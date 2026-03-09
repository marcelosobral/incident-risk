from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.features.event_features import build_event_features, make_event_table
from src.features.labels import build_rth_events
from src.utils.time_utils import pick_first_col, pick_time_col_for_table


@dataclass
class ComorbidityRule:
    name: str
    icd_prefixes: tuple[str, ...]


COMORBIDITY_RULES = [
    ComorbidityRule("dementia", ("F01", "F02", "F03", "G30")),
    ComorbidityRule("chf", ("I50",)),
    ComorbidityRule("copd", ("J44",)),
    ComorbidityRule("diabetes", ("E10", "E11", "E13")),
    ComorbidityRule("ckd", ("N18",)),
    ComorbidityRule("stroke", ("I61", "I63", "I64")),
    ComorbidityRule("parkinsons", ("G20",)),
    ComorbidityRule("osteoporosis", ("M81",)),
    ComorbidityRule("depression", ("F32", "F33")),
    ComorbidityRule("delirium", ("F05",)),
]

VITAL_TYPE_MAP = {
    "pain level": "pain_level",
    "bp - systolic": "bp_systolic",
    "pulse": "pulse",
    "o2 sats": "o2_sats",
    "blood sugar": "blood_sugar",
    "temperature": "temperature",
    "respiration": "respiration",
    "weight": "weight",
}

VITAL_ABNORMAL_RULES = {
    "temperature": ("gte", 100.4),
    "pulse": ("gte", 100.0),
    "o2_sats": ("lte", 92.0),
    "blood_sugar": ("range", 70.0, 200.0),
    "bp_systolic": ("range", 90.0, 160.0),
    "respiration": ("gte", 24.0),
    "pain_level": ("gte", 7.0),
}

MED_RISK_KEYWORDS = {
    "anticoagulant": ["warfarin", "apixaban", "rivaroxaban", "dabigatran", "heparin"],
    "sedative": ["benzodiazepine", "lorazepam", "diazepam", "alprazolam", "midazolam"],
    "opioid": ["morphine", "oxycodone", "hydrocodone", "fentanyl", "tramadol"],
    "antipsychotic": ["quetiapine", "risperidone", "haloperidol", "olanzapine", "clozapine"],
}


GG_TASKS = {
    "sit to stand",
    "toilet transfer",
    "chair / bed-to-chair transfer",
    "chair/bed-to-chair transfer",
    "walk 10 feet",
    "lower body dressing",
    "putting on / taking off footwear",
    "putting on/taking off footwear",
    "lying to sitting on side of bed",
    "picking up object",
    "roll left and right",
}

ASSIST_SCORE_MAP = {
    "independent": 0,
    "setup": 1,
    "set up": 1,
    "supervision": 2,
    "partial": 3,
    "substantial": 4,
    "dependent": 5,
    "not applicable": 5,
    "safety": 5,
}

HIGH_RISK_DX_CODES = (
    "M62.81",
    "R26.2",
    "R26.81",
    "R27.8",
    "R27.9",
    "M62.59",
    "R41.841",
    "F03.90",
    "G30.9",
    "Z74.1",
    "Z91.81",
    "W19.XXXD",
)


def add_resident_demographics(
    snapshots: pd.DataFrame,
    residents: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    work = snapshots.copy() if copy else snapshots
    if residents.empty:
        return work

    dob_col = pick_first_col(residents, ["date_of_birth", "dob", "birth_date", "birthdate"])
    sex_col = pick_first_col(residents, ["sex", "gender"])
    admit_col = pick_first_col(residents, ["admission_date", "admit_date", "admitted_at", "admit_at", "start_at"])
    outpatient_col = pick_first_col(residents, ["outpatient"])

    cols = ["resident_id"]
    for c in [dob_col, sex_col, admit_col, outpatient_col]:
        if c and c not in cols:
            cols.append(c)

    demo = residents[cols].copy()
    if dob_col:
        demo["dob_ts"] = pd.to_datetime(demo[dob_col], errors="coerce")
    if admit_col:
        demo["admit_ts"] = pd.to_datetime(demo[admit_col], errors="coerce")

    work = work.merge(demo, on="resident_id", how="left")

    if dob_col:
        work["age"] = (work["snapshot_date"] - work["dob_ts"]).dt.days / 365.25
    if admit_col:
        work["days_since_admission"] = (work["snapshot_date"] - work["admit_ts"]).dt.days
        bins = [-1, 30, 90, 180, 365, np.inf]
        labels = ["los_0_30", "los_31_90", "los_91_180", "los_181_365", "los_366_plus"]
        los_bucket = pd.cut(work["days_since_admission"], bins=bins, labels=labels)
        for label in labels:
            work[label] = (los_bucket == label).astype(int)

    if sex_col:
        work["sex"] = work[sex_col].astype(str)
    if outpatient_col:
        work["outpatient_flag"] = work[outpatient_col].astype(float)

    drop_cols = [c for c in [dob_col, sex_col, admit_col, outpatient_col, "dob_ts", "admit_ts"] if c in work.columns]
    if drop_cols:
        work = work.drop(columns=drop_cols)

    return work


def add_comorbidity_flags(
    snapshots: pd.DataFrame,
    diagnoses: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    work = snapshots.copy() if copy else snapshots
    if diagnoses.empty:
        for rule in COMORBIDITY_RULES:
            work[f"comorb_{rule.name}"] = 0
        work["comorbidity_count"] = 0
        return work

    code_col = pick_first_col(diagnoses, ["icd_10_code", "icd10_code", "diagnosis_code", "code"])
    time_col = pick_time_col_for_table("diagnoses", diagnoses)
    if code_col is None or time_col is None or "resident_id" not in diagnoses.columns:
        for rule in COMORBIDITY_RULES:
            work[f"comorb_{rule.name}"] = 0
        work["comorbidity_count"] = 0
        return work

    dx = diagnoses[["resident_id", code_col, time_col]].copy()
    dx[code_col] = dx[code_col].astype(str).str.upper()
    dx["event_time"] = pd.to_datetime(dx[time_col], errors="coerce")
    dx = dx[dx["resident_id"].notna() & dx["event_time"].notna()]

    work = work.sort_values(["resident_id", "snapshot_date"]).reset_index(drop=True)

    for rule in COMORBIDITY_RULES:
        work[f"comorb_{rule.name}"] = 0

    dx_by_resident = {
        resident_id: group
        for resident_id, group in dx.groupby("resident_id", sort=False)
    }

    for resident_id, snap_group in work.groupby("resident_id", sort=False):
        dx_group = dx_by_resident.get(resident_id)
        if dx_group is None or dx_group.empty:
            continue
        snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
        for rule in COMORBIDITY_RULES:
            rule_mask = dx_group[code_col].str.startswith(rule.icd_prefixes)
            rule_times = dx_group.loc[rule_mask, "event_time"].sort_values().to_numpy(dtype="datetime64[ns]")
            if rule_times.size == 0:
                continue
            right_idx = np.searchsorted(rule_times, snap_times, side="right")
            work.loc[snap_group.index, f"comorb_{rule.name}"] = (right_idx > 0).astype(int)

    comorb_cols = [f"comorb_{rule.name}" for rule in COMORBIDITY_RULES]
    work["comorbidity_count"] = work[comorb_cols].sum(axis=1)
    return work


def _trend_features_for_table(
    snapshots: pd.DataFrame,
    table: pd.DataFrame,
    table_name: str,
    value_candidates: Iterable[str],
    prefix: str,
    windows_days: Iterable[int] = (30, 90),
    copy: bool = True,
) -> pd.DataFrame:
    work = snapshots.copy() if copy else snapshots
    if table.empty:
        return work

    time_col = pick_time_col_for_table(table_name, table)
    if time_col is None or "resident_id" not in table.columns:
        return work

    value_col = pick_first_col(table, value_candidates)
    if value_col is None:
        return work

    data = table[["resident_id", time_col, value_col]].copy()
    data["event_time"] = pd.to_datetime(data[time_col], errors="coerce")
    data["value"] = pd.to_numeric(data[value_col], errors="coerce")
    data = data[data["resident_id"].notna() & data["event_time"].notna() & data["value"].notna()]
    if data.empty:
        return work

    work = work.sort_values(["resident_id", "snapshot_date"]).reset_index(drop=True)

    last_col = f"{prefix}_last"
    work[last_col] = np.nan
    for w in windows_days:
        work[f"{prefix}_mean_{w}d"] = np.nan
        work[f"{prefix}_std_{w}d"] = np.nan
        work[f"{prefix}_slope_{w}d"] = np.nan
        work[f"{prefix}_decline_{w}d"] = 0

    data_by_resident = {
        resident_id: group.sort_values("event_time")
        for resident_id, group in data.groupby("resident_id", sort=False)
    }

    for resident_id, snap_group in work.groupby("resident_id", sort=False):
        ev_group = data_by_resident.get(resident_id)
        if ev_group is None or ev_group.empty:
            continue
        ev_times = ev_group["event_time"].to_numpy(dtype="datetime64[ns]")
        ev_vals = ev_group["value"].to_numpy(dtype=float)
        snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")

        right_idx = np.searchsorted(ev_times, snap_times, side="right")
        has_event = right_idx > 0
        last_idx = right_idx[has_event] - 1
        last_vals = ev_vals[last_idx]
        work.loc[snap_group.index[has_event], last_col] = last_vals

        for w in windows_days:
            left_idx = np.searchsorted(ev_times, snap_times - np.timedelta64(w, "D"), side="right")
            mean_vals = np.full(len(snap_times), np.nan)
            std_vals = np.full(len(snap_times), np.nan)
            slope_vals = np.full(len(snap_times), np.nan)
            decline_vals = np.zeros(len(snap_times), dtype=int)
            for i, (l_idx, r_idx) in enumerate(zip(left_idx, right_idx)):
                if r_idx - l_idx <= 0:
                    continue
                vals = ev_vals[l_idx:r_idx]
                times = ev_times[l_idx:r_idx]
                mean_vals[i] = float(np.mean(vals))
                std_vals[i] = float(np.std(vals))
                if len(vals) >= 2:
                    delta_days = (times[-1] - times[0]).astype("timedelta64[D]").astype(float)
                    slope = (vals[-1] - vals[0]) / delta_days if delta_days > 0 else 0.0
                    slope_vals[i] = float(slope)
                    decline_vals[i] = int(vals[-1] < vals[0])

            work.loc[snap_group.index, f"{prefix}_mean_{w}d"] = mean_vals
            work.loc[snap_group.index, f"{prefix}_std_{w}d"] = std_vals
            work.loc[snap_group.index, f"{prefix}_slope_{w}d"] = slope_vals
            work.loc[snap_group.index, f"{prefix}_decline_{w}d"] = decline_vals

    return work


def add_functional_status_trends(
    snapshots: pd.DataFrame,
    adl_responses: pd.DataFrame,
    gg_responses: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    work = snapshots.copy() if copy else snapshots
    work = _trend_features_for_table(
        work,
        adl_responses,
        table_name="adl_responses",
        value_candidates=["response_code", "response", "adl_change"],
        prefix="adl",
        copy=False,
    )
    work = _trend_features_for_table(
        work,
        gg_responses,
        table_name="gg_responses",
        value_candidates=["response_code", "response", "change"],
        prefix="gg",
        copy=False,
    )
    return work


def add_vital_type_features(
    snapshots: pd.DataFrame,
    vitals: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    work = snapshots.copy() if copy else snapshots
    if vitals.empty or "resident_id" not in vitals.columns:
        return work

    time_col = pick_time_col_for_table("vitals", vitals)
    if time_col is None or "vital_type" not in vitals.columns:
        return work

    value_col = pick_first_col(vitals, ["value", "systolic_value", "dystolic_value"])
    if value_col is None:
        return work

    data = vitals[["resident_id", "vital_type", time_col, value_col]].copy()
    data["event_time"] = pd.to_datetime(data[time_col], errors="coerce")
    data["value"] = pd.to_numeric(data[value_col], errors="coerce")
    data["type_key"] = data["vital_type"].astype(str).str.lower().map(VITAL_TYPE_MAP)
    data = data[data["resident_id"].notna() & data["event_time"].notna() & data["value"].notna()]
    data = data[data["type_key"].notna()]
    if data.empty:
        return work

    work = work.sort_values(["resident_id", "snapshot_date"]).reset_index(drop=True)

    for type_key in sorted(data["type_key"].unique()):
        prefix = f"vitals_{type_key}"
        work[f"{prefix}_last"] = np.nan
        for w in (30, 90):
            work[f"{prefix}_mean_{w}d"] = np.nan
            work[f"{prefix}_min_{w}d"] = np.nan
            work[f"{prefix}_max_{w}d"] = np.nan
            work[f"{prefix}_std_{w}d"] = np.nan
        if type_key in VITAL_ABNORMAL_RULES:
            work[f"{prefix}_abnormal_30d"] = 0

    data_by_resident = {
        resident_id: group
        for resident_id, group in data.groupby("resident_id", sort=False)
    }

    for resident_id, snap_group in work.groupby("resident_id", sort=False):
        v_group = data_by_resident.get(resident_id)
        if v_group is None or v_group.empty:
            continue
        snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
        for type_key, type_group in v_group.groupby("type_key"):
            type_group = type_group.sort_values("event_time")
            ev_times = type_group["event_time"].to_numpy(dtype="datetime64[ns]")
            ev_vals = type_group["value"].to_numpy(dtype=float)
            prefix = f"vitals_{type_key}"

            right_idx = np.searchsorted(ev_times, snap_times, side="right")
            has_event = right_idx > 0
            last_vals = np.full(len(snap_times), np.nan)
            if np.any(has_event):
                last_vals[has_event] = ev_vals[right_idx[has_event] - 1]
            work.loc[snap_group.index, f"{prefix}_last"] = last_vals

            for w in (30, 90):
                left_idx = np.searchsorted(ev_times, snap_times - np.timedelta64(w, "D"), side="right")
                mean_vals = np.full(len(snap_times), np.nan)
                min_vals = np.full(len(snap_times), np.nan)
                max_vals = np.full(len(snap_times), np.nan)
                std_vals = np.full(len(snap_times), np.nan)
                for i, (l_idx, r_idx) in enumerate(zip(left_idx, right_idx)):
                    if r_idx - l_idx <= 0:
                        continue
                    vals = ev_vals[l_idx:r_idx]
                    mean_vals[i] = float(np.mean(vals))
                    min_vals[i] = float(np.min(vals))
                    max_vals[i] = float(np.max(vals))
                    std_vals[i] = float(np.std(vals))

                work.loc[snap_group.index, f"{prefix}_mean_{w}d"] = mean_vals
                work.loc[snap_group.index, f"{prefix}_min_{w}d"] = min_vals
                work.loc[snap_group.index, f"{prefix}_max_{w}d"] = max_vals
                work.loc[snap_group.index, f"{prefix}_std_{w}d"] = std_vals

            if type_key in VITAL_ABNORMAL_RULES:
                rule = VITAL_ABNORMAL_RULES[type_key]
                left_30 = np.searchsorted(ev_times, snap_times - np.timedelta64(30, "D"), side="right")
                abnormal_vals = np.zeros(len(snap_times), dtype=int)
                for i, (l_idx, r_idx) in enumerate(zip(left_30, right_idx)):
                    if r_idx - l_idx <= 0:
                        continue
                    vals = ev_vals[l_idx:r_idx]
                    if rule[0] == "gte":
                        abnormal_vals[i] = int(np.nanmax(vals) >= rule[1])
                    elif rule[0] == "lte":
                        abnormal_vals[i] = int(np.nanmin(vals) <= rule[1])
                    elif rule[0] == "range":
                        low, high = rule[1], rule[2]
                        abnormal_vals[i] = int((np.nanmin(vals) <= low) or (np.nanmax(vals) >= high))

                work.loc[snap_group.index, f"{prefix}_abnormal_30d"] = abnormal_vals

    if "vitals_weight_last" in work.columns:
        work["last_weight"] = work["vitals_weight_last"]

    return work


def add_medication_risk(
    snapshots: pd.DataFrame,
    medications: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    work = snapshots.copy() if copy else snapshots
    if medications.empty or "resident_id" not in medications.columns:
        return work

    time_col = pick_time_col_for_table("medications", medications)
    if time_col is None:
        return work

    name_col = pick_first_col(
        medications,
        ["description", "medication_name", "name", "drug_name", "order_name"],
    )
    if name_col is None:
        return work

    data = medications[["resident_id", time_col, name_col]].copy()
    data["event_time"] = pd.to_datetime(data[time_col], errors="coerce")
    data["med_name"] = data[name_col].astype(str).str.lower()
    data = data[data["resident_id"].notna() & data["event_time"].notna()]
    if data.empty:
        return work

    work = work.sort_values(["resident_id", "snapshot_date"]).reset_index(drop=True)
    work["polypharmacy_90d"] = 0
    work["med_events_30d"] = 0
    work["med_new_30d"] = 0
    for key in MED_RISK_KEYWORDS:
        work[f"med_risk_{key}_90d"] = 0

    data_by_resident = {
        resident_id: group.sort_values("event_time")
        for resident_id, group in data.groupby("resident_id", sort=False)
    }
    first_times = data.groupby(["resident_id", "med_name"], sort=False)["event_time"].min().reset_index()
    first_times_by_resident = {
        resident_id: group.sort_values("event_time")
        for resident_id, group in first_times.groupby("resident_id", sort=False)
    }

    for resident_id, snap_group in work.groupby("resident_id", sort=False):
        med_group = data_by_resident.get(resident_id)
        if med_group is None or med_group.empty:
            continue
        snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
        med_times = med_group["event_time"].to_numpy(dtype="datetime64[ns]")
        med_names = med_group["med_name"].to_numpy()

        right_idx = np.searchsorted(med_times, snap_times, side="right")
        left_90 = np.searchsorted(med_times, snap_times - np.timedelta64(90, "D"), side="right")
        left_30 = np.searchsorted(med_times, snap_times - np.timedelta64(30, "D"), side="right")

        first_group = first_times_by_resident.get(resident_id)
        first_vals = (
            first_group["event_time"].to_numpy(dtype="datetime64[ns]")
            if first_group is not None
            else np.array([], dtype="datetime64[ns]")
        )

        poly_vals = np.zeros(len(snap_times), dtype=int)
        events_30 = np.zeros(len(snap_times), dtype=int)
        new_30 = np.zeros(len(snap_times), dtype=int)
        risk_vals = {key: np.zeros(len(snap_times), dtype=int) for key in MED_RISK_KEYWORDS}

        for i, (l90, l30, r_idx) in enumerate(zip(left_90, left_30, right_idx)):
            window_names = med_names[l90:r_idx]
            if window_names.size:
                poly_vals[i] = int(len(set(window_names)))
                for key, tokens in MED_RISK_KEYWORDS.items():
                    risk_vals[key][i] = int(any(t in name for name in window_names for t in tokens))
            events_30[i] = int(max(0, r_idx - l30))

            if first_vals.size:
                new_left = np.searchsorted(first_vals, snap_times[i] - np.timedelta64(30, "D"), side="right")
                new_right = np.searchsorted(first_vals, snap_times[i], side="right")
                new_30[i] = int(max(0, new_right - new_left))

        work.loc[snap_group.index, "polypharmacy_90d"] = poly_vals
        work.loc[snap_group.index, "med_events_30d"] = events_30
        work.loc[snap_group.index, "med_new_30d"] = new_30
        for key, values in risk_vals.items():
            work.loc[snap_group.index, f"med_risk_{key}_90d"] = values

    return work


def add_fall_injury_features(
    snapshots: pd.DataFrame,
    incidents: pd.DataFrame,
    injuries: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    work = snapshots.copy() if copy else snapshots
    if incidents.empty or injuries.empty:
        return work

    if "incident_id" not in incidents.columns or "incident_id" not in injuries.columns:
        return work

    inc = incidents.copy()
    if "incident_type" not in inc.columns:
        return work
    inc["incident_type"] = inc["incident_type"].astype(str)
    fall_mask = inc["incident_type"].str.lower().str.contains("fall", na=False)
    inc = inc[fall_mask]
    time_col = pick_first_col(inc, ["occurred_at", "created_at"])
    if time_col is None:
        return work

    inj = injuries[["incident_id"]].dropna()
    inc = inc.merge(inj, on="incident_id", how="inner")
    if inc.empty:
        return work

    events = make_event_table(inc, time_col)
    work = build_event_features(work, events, windows_days=[90, 180], prefix="fall_injury")
    return work


def add_care_plan_signals(
    snapshots: pd.DataFrame,
    care_plans: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    work = snapshots.copy() if copy else snapshots
    if care_plans.empty:
        work["care_plans_count_total"] = 0
        return work

    time_col = pick_time_col_for_table("care_plans", care_plans)
    if time_col is None:
        work["care_plans_count_total"] = 0
        return work

    events = make_event_table(care_plans, time_col)
    work = build_event_features(work, events, windows_days=[], prefix="care_plans")

    work["care_plans_count_total"] = 0
    events = events.sort_values(["resident_id", "event_time"])
    for resident_id, snap_group in work.groupby("resident_id", sort=False):
        ev_group = events[events["resident_id"] == resident_id]
        if ev_group.empty:
            continue
        ev_times = ev_group["event_time"].to_numpy(dtype="datetime64[ns]")
        snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
        right_idx = np.searchsorted(ev_times, snap_times, side="right")
        work.loc[snap_group.index, "care_plans_count_total"] = right_idx

    return work


def add_facility_effects(
    snapshots: pd.DataFrame,
    incidents: pd.DataFrame,
    admissions: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    work = snapshots.copy() if copy else snapshots
    if "facility_id" not in work.columns:
        return work

    work = work.sort_values(["facility_id", "snapshot_date"]).reset_index(drop=True)

    def build_facility_events(df: pd.DataFrame, time_candidates: list[str]) -> pd.DataFrame:
        if df.empty or "facility_id" not in df.columns:
            return pd.DataFrame(columns=["facility_id", "event_time"])
        time_col = pick_first_col(df, time_candidates)
        if time_col is None:
            return pd.DataFrame(columns=["facility_id", "event_time"])
        events = df[["facility_id", time_col]].copy()
        events["event_time"] = pd.to_datetime(events[time_col], errors="coerce")
        events = events[events["facility_id"].notna() & events["event_time"].notna()]
        return events[["facility_id", "event_time"]]

    fall_events = incidents.copy()
    if "incident_type" in fall_events.columns:
        fall_events["incident_type"] = fall_events["incident_type"].astype(str)
        fall_events = fall_events[fall_events["incident_type"].str.lower().str.contains("fall", na=False)]
    fall_events = build_facility_events(fall_events, ["occurred_at", "created_at"])

    rth_events = build_facility_events(
        admissions,
        ["effective_date", "admission_date", "admit_date", "admitted_at", "admit_at", "start_at"],
    )

    work["facility_fall_90d"] = 0
    work["facility_rth_90d"] = 0

    for facility_id, snap_group in work.groupby("facility_id", sort=False):
        snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
        fall_group = fall_events[fall_events["facility_id"] == facility_id]
        rth_group = rth_events[rth_events["facility_id"] == facility_id]

        if not fall_group.empty:
            fall_times = fall_group["event_time"].sort_values().to_numpy(dtype="datetime64[ns]")
            right_idx = np.searchsorted(fall_times, snap_times, side="right")
            left_idx = np.searchsorted(fall_times, snap_times - np.timedelta64(90, "D"), side="right")
            work.loc[snap_group.index, "facility_fall_90d"] = right_idx - left_idx

        if not rth_group.empty:
            rth_times = rth_group["event_time"].sort_values().to_numpy(dtype="datetime64[ns]")
            right_idx = np.searchsorted(rth_times, snap_times, side="right")
            left_idx = np.searchsorted(rth_times, snap_times - np.timedelta64(90, "D"), side="right")
            work.loc[snap_group.index, "facility_rth_90d"] = right_idx - left_idx

    facility_counts = work.groupby(["facility_id", "snapshot_date"]).size().rename("facility_residents")
    work = work.merge(facility_counts, on=["facility_id", "snapshot_date"], how="left")
    work["facility_fall_rate_90d"] = work["facility_fall_90d"] / work["facility_residents"].replace(0, np.nan)
    work["facility_rth_rate_90d"] = work["facility_rth_90d"] / work["facility_residents"].replace(0, np.nan)
    return work


def _map_assist_score(value: str) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().lower()
    for key, score in ASSIST_SCORE_MAP.items():
        if key in text:
            return float(score)
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric):
        numeric = float(numeric)
        if numeric in {0, 1, 2, 3, 4, 5}:
            return numeric
        if numeric == 9:
            return 5.0
    return np.nan


def add_resident_risk_signals(
    snapshots: pd.DataFrame,
    gg_responses: pd.DataFrame,
    medications: pd.DataFrame,
    adl_responses: pd.DataFrame,
    diagnoses: pd.DataFrame,
    vitals: pd.DataFrame,
    admissions: pd.DataFrame,
    incidents: pd.DataFrame,
    therapy_tracks: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    work = snapshots.copy() if copy else snapshots
    if work.empty or "resident_id" not in work.columns:
        return work

    work = work.sort_values(["resident_id", "snapshot_date"]).reset_index(drop=True)

    work["gg_mobility_mean"] = np.nan
    work["gg_mobility_latest_sum"] = np.nan
    work["polypharmacy_30d_count"] = 0
    work["polypharmacy_30d_flag"] = 0
    work["gait_instability_flag"] = 0
    work["gait_assist_change_3"] = np.nan
    work["diagnosis_high_risk_flag"] = 0
    work["bp_systolic_std_24h"] = np.nan
    work["bp_systolic_max_drop_24h"] = np.nan
    work["bp_systolic_variance_flag"] = 0
    work["pain_level_mean_7d"] = np.nan
    work["pain_level_severe_latest"] = 0
    work["recent_rth_60d"] = 0
    work["days_since_last_fall"] = np.nan
    work["fall_frequency_180d"] = 0
    work["pt_active_flag"] = 0
    work["pt_done_before_flag"] = 0

    if not gg_responses.empty:
        time_col = pick_time_col_for_table("gg_responses", gg_responses)
        task_col = pick_first_col(
            gg_responses,
            ["task_name", "task", "activity", "item", "question", "gg_item", "gg_task"],
        )
        resp_col = pick_first_col(
            gg_responses,
            ["response_text", "response_description", "response", "response_code", "score"],
        )
        if time_col and task_col and resp_col and "resident_id" in gg_responses.columns:
            gg = gg_responses[["resident_id", time_col, task_col, resp_col]].copy()
            gg["event_time"] = pd.to_datetime(gg[time_col], errors="coerce")
            gg["task"] = gg[task_col].astype(str).str.lower().str.strip()
            gg = gg[gg["task"].isin(GG_TASKS)]
            gg["score"] = gg[resp_col].apply(_map_assist_score)
            gg = gg[gg["resident_id"].notna() & gg["event_time"].notna() & gg["score"].notna()]
            gg_by_resident = {
                resident_id: group
                for resident_id, group in gg.groupby("resident_id", sort=False)
            }

            for resident_id, snap_group in work.groupby("resident_id", sort=False):
                gg_group = gg_by_resident.get(resident_id)
                if gg_group is None or gg_group.empty:
                    continue
                snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
                task_scores = []
                for task in GG_TASKS:
                    t_group = gg_group[gg_group["task"] == task].sort_values("event_time")
                    if t_group.empty:
                        task_scores.append(np.full(len(snap_times), np.nan))
                        continue
                    times = t_group["event_time"].to_numpy(dtype="datetime64[ns]")
                    scores = t_group["score"].to_numpy(dtype=float)
                    right_idx = np.searchsorted(times, snap_times, side="right")
                    has_event = right_idx > 0
                    latest = np.full(len(snap_times), np.nan)
                    latest[has_event] = scores[right_idx[has_event] - 1]
                    task_scores.append(latest)

                score_matrix = np.vstack(task_scores) if task_scores else np.empty((0, len(snap_times)))
                latest_sum = np.nansum(score_matrix, axis=0) if score_matrix.size else np.nan
                latest_mean = np.nanmean(score_matrix, axis=0) if score_matrix.size else np.nan
                work.loc[snap_group.index, "gg_mobility_mean"] = latest_mean
                work.loc[snap_group.index, "gg_mobility_latest_sum"] = latest_sum

    if not medications.empty:
        time_col = pick_time_col_for_table("medications", medications)
        status_col = pick_first_col(medications, ["status", "administration_status", "med_status"])
        desc_col = pick_first_col(
            medications,
            ["description", "medication_name", "name", "drug_name", "order_name"],
        )
        if time_col and status_col and desc_col and "resident_id" in medications.columns:
            meds = medications[["resident_id", time_col, status_col, desc_col]].copy()
            meds["event_time"] = pd.to_datetime(meds[time_col], errors="coerce")
            meds["status"] = meds[status_col].astype(str).str.lower()
            meds["desc"] = meds[desc_col].astype(str)
            meds = meds[meds["resident_id"].notna() & meds["event_time"].notna()]
            status_mask = meds["status"].isin(["on time", "late"])
            meds = meds[status_mask]
            meds_by_resident = {
                resident_id: group.sort_values("event_time")
                for resident_id, group in meds.groupby("resident_id", sort=False)
            }

            for resident_id, snap_group in work.groupby("resident_id", sort=False):
                med_group = meds_by_resident.get(resident_id)
                if med_group is None or med_group.empty:
                    continue
                snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
                med_times = med_group["event_time"].to_numpy(dtype="datetime64[ns]")
                med_desc = med_group["desc"].to_numpy(dtype=object)
                right_idx = np.searchsorted(med_times, snap_times, side="right")
                left_30 = np.searchsorted(med_times, snap_times - np.timedelta64(30, "D"), side="right")

                poly_count = np.zeros(len(snap_times), dtype=int)
                poly_flag = np.zeros(len(snap_times), dtype=int)
                for i, (l30, r_idx) in enumerate(zip(left_30, right_idx)):
                    window_desc = med_desc[l30:r_idx]
                    count = int(len(set(window_desc))) if window_desc.size else 0
                    poly_count[i] = count
                    poly_flag[i] = int(count >= 5)

                work.loc[snap_group.index, "polypharmacy_30d_count"] = poly_count
                work.loc[snap_group.index, "polypharmacy_30d_flag"] = poly_flag

    if not adl_responses.empty:
        time_col = pick_time_col_for_table("adl_responses", adl_responses)
        activity_col = pick_first_col(adl_responses, ["activity", "task", "question", "item"])
        resp_col = pick_first_col(
            adl_responses,
            ["response_description", "response", "response_text", "response_code", "adl_change"],
        )
        if time_col and activity_col and resp_col and "resident_id" in adl_responses.columns:
            adl = adl_responses[["resident_id", time_col, activity_col, resp_col]].copy()
            adl["event_time"] = pd.to_datetime(adl[time_col], errors="coerce")
            adl["activity"] = adl[activity_col].astype(str).str.lower()
            adl = adl[adl["activity"].str.contains("walking|gait", na=False)]
            adl["score"] = adl[resp_col].apply(_map_assist_score)
            adl = adl[adl["resident_id"].notna() & adl["event_time"].notna() & adl["score"].notna()]
            adl_by_resident = {
                resident_id: group.sort_values("event_time")
                for resident_id, group in adl.groupby("resident_id", sort=False)
            }

            for resident_id, snap_group in work.groupby("resident_id", sort=False):
                adl_group = adl_by_resident.get(resident_id)
                if adl_group is None or adl_group.empty:
                    continue
                snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
                adl_times = adl_group["event_time"].to_numpy(dtype="datetime64[ns]")
                adl_scores = adl_group["score"].to_numpy(dtype=float)
                right_idx = np.searchsorted(adl_times, snap_times, side="right")

                assist_change = np.full(len(snap_times), np.nan)
                instability = np.zeros(len(snap_times), dtype=int)
                for i, r_idx in enumerate(right_idx):
                    if r_idx <= 0:
                        continue
                    start = max(0, r_idx - 3)
                    window = adl_scores[start:r_idx]
                    if window.size < 2:
                        continue
                    change = float(window[-1] - window[0])
                    assist_change[i] = change
                    instability[i] = int(window[0] == 0 and window[-1] > 0)

                work.loc[snap_group.index, "gait_assist_change_3"] = assist_change
                work.loc[snap_group.index, "gait_instability_flag"] = instability

    if not diagnoses.empty:
        code_col = pick_first_col(diagnoses, ["icd_10_code", "icd10_code", "diagnosis_code", "code"])
        time_col = pick_time_col_for_table("diagnoses", diagnoses)
        if code_col and time_col and "resident_id" in diagnoses.columns:
            dx = diagnoses[["resident_id", code_col, time_col]].copy()
            dx["event_time"] = pd.to_datetime(dx[time_col], errors="coerce")
            dx["code"] = dx[code_col].astype(str).str.upper()
            dx = dx[dx["resident_id"].notna() & dx["event_time"].notna()]
            dx = dx[dx["code"].str.startswith(HIGH_RISK_DX_CODES)]
            dx_by_resident = {
                resident_id: group.sort_values("event_time")
                for resident_id, group in dx.groupby("resident_id", sort=False)
            }

            for resident_id, snap_group in work.groupby("resident_id", sort=False):
                dx_group = dx_by_resident.get(resident_id)
                if dx_group is None or dx_group.empty:
                    continue
                snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
                dx_times = dx_group["event_time"].to_numpy(dtype="datetime64[ns]")
                right_idx = np.searchsorted(dx_times, snap_times, side="right")
                work.loc[snap_group.index, "diagnosis_high_risk_flag"] = (right_idx > 0).astype(int)

    if not vitals.empty:
        time_col = pick_time_col_for_table("vitals", vitals)
        value_col = pick_first_col(vitals, ["value", "systolic_value", "dystolic_value"])
        if time_col and value_col and "resident_id" in vitals.columns and "vital_type" in vitals.columns:
            vit = vitals[["resident_id", "vital_type", time_col, value_col]].copy()
            vit["event_time"] = pd.to_datetime(vit[time_col], errors="coerce")
            vit["value"] = pd.to_numeric(vit[value_col], errors="coerce")
            vit = vit[vit["resident_id"].notna() & vit["event_time"].notna() & vit["value"].notna()]
            vit["type_key"] = vit["vital_type"].astype(str).str.lower()
            bp = vit[vit["type_key"] == "bp - systolic"]
            pain = vit[vit["type_key"] == "pain level"]

            bp_by_resident = {
                resident_id: group.sort_values("event_time")
                for resident_id, group in bp.groupby("resident_id", sort=False)
            }
            pain_by_resident = {
                resident_id: group.sort_values("event_time")
                for resident_id, group in pain.groupby("resident_id", sort=False)
            }

            for resident_id, snap_group in work.groupby("resident_id", sort=False):
                snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")

                bp_group = bp_by_resident.get(resident_id)
                if bp_group is not None and not bp_group.empty:
                    bp_times = bp_group["event_time"].to_numpy(dtype="datetime64[ns]")
                    bp_vals = bp_group["value"].to_numpy(dtype=float)
                    right_idx = np.searchsorted(bp_times, snap_times, side="right")
                    left_24 = np.searchsorted(bp_times, snap_times - np.timedelta64(1, "D"), side="right")

                    bp_std = np.full(len(snap_times), np.nan)
                    bp_drop = np.full(len(snap_times), np.nan)
                    bp_flag = np.zeros(len(snap_times), dtype=int)
                    for i, (l_idx, r_idx) in enumerate(zip(left_24, right_idx)):
                        if r_idx - l_idx <= 0:
                            continue
                        vals = bp_vals[l_idx:r_idx]
                        std_val = float(np.std(vals))
                        bp_std[i] = std_val
                        bp_drop[i] = float(np.max(vals) - np.min(vals))
                        bp_flag[i] = int(std_val > 20.0)

                    work.loc[snap_group.index, "bp_systolic_std_24h"] = bp_std
                    work.loc[snap_group.index, "bp_systolic_max_drop_24h"] = bp_drop
                    work.loc[snap_group.index, "bp_systolic_variance_flag"] = bp_flag

                pain_group = pain_by_resident.get(resident_id)
                if pain_group is not None and not pain_group.empty:
                    pain_times = pain_group["event_time"].to_numpy(dtype="datetime64[ns]")
                    pain_vals = pain_group["value"].to_numpy(dtype=float)
                    right_idx = np.searchsorted(pain_times, snap_times, side="right")
                    left_7 = np.searchsorted(pain_times, snap_times - np.timedelta64(7, "D"), side="right")

                    pain_mean = np.full(len(snap_times), np.nan)
                    pain_severe = np.zeros(len(snap_times), dtype=int)
                    for i, (l_idx, r_idx) in enumerate(zip(left_7, right_idx)):
                        if r_idx - l_idx <= 0:
                            continue
                        vals = pain_vals[l_idx:r_idx]
                        pain_mean[i] = float(np.mean(vals))
                        latest_val = pain_vals[r_idx - 1] if r_idx > 0 else np.nan
                        pain_severe[i] = int(latest_val >= 7)

                    work.loc[snap_group.index, "pain_level_mean_7d"] = pain_mean
                    work.loc[snap_group.index, "pain_level_severe_latest"] = pain_severe

    if not admissions.empty:
        rth_events = build_rth_events(admissions)
        if not rth_events.empty:
            rth_events = rth_events.sort_values(["resident_id", "event_time"])
            rth_by_resident = {
                resident_id: group
                for resident_id, group in rth_events.groupby("resident_id", sort=False)
            }
            for resident_id, snap_group in work.groupby("resident_id", sort=False):
                ev_group = rth_by_resident.get(resident_id)
                if ev_group is None or ev_group.empty:
                    continue
                ev_times = ev_group["event_time"].to_numpy(dtype="datetime64[ns]")
                snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
                right_idx = np.searchsorted(ev_times, snap_times, side="right")
                left_idx = np.searchsorted(ev_times, snap_times - np.timedelta64(60, "D"), side="right")
                work.loc[snap_group.index, "recent_rth_60d"] = (right_idx > left_idx).astype(int)

    if not incidents.empty and "incident_type" in incidents.columns:
        inc = incidents.copy()
        inc["incident_type"] = inc["incident_type"].astype(str)
        inc = inc[inc["incident_type"].str.lower().str.contains("fall", na=False)]
        time_col = pick_first_col(inc, ["occurred_at", "created_at"])
        if time_col and "resident_id" in inc.columns:
            inc["event_time"] = pd.to_datetime(inc[time_col], errors="coerce")
            inc = inc[inc["resident_id"].notna() & inc["event_time"].notna()]
            inc = inc.sort_values(["resident_id", "event_time"])
            inc_by_resident = {
                resident_id: group
                for resident_id, group in inc.groupby("resident_id", sort=False)
            }
            for resident_id, snap_group in work.groupby("resident_id", sort=False):
                inc_group = inc_by_resident.get(resident_id)
                if inc_group is None or inc_group.empty:
                    continue
                ev_times = inc_group["event_time"].to_numpy(dtype="datetime64[ns]")
                snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
                right_idx = np.searchsorted(ev_times, snap_times, side="right")
                left_180 = np.searchsorted(ev_times, snap_times - np.timedelta64(180, "D"), side="right")

                days_since = np.full(len(snap_times), np.nan)
                fall_freq = np.zeros(len(snap_times), dtype=int)
                for i, (l_idx, r_idx) in enumerate(zip(left_180, right_idx)):
                    if r_idx > 0:
                        last_time = ev_times[r_idx - 1]
                        days_since[i] = float((snap_times[i] - last_time) / np.timedelta64(1, "D"))
                    fall_freq[i] = int(max(0, r_idx - l_idx))

                work.loc[snap_group.index, "days_since_last_fall"] = days_since
                work.loc[snap_group.index, "fall_frequency_180d"] = fall_freq

    if not therapy_tracks.empty:
        if "resident_id" in therapy_tracks.columns:
            discipline_col = pick_first_col(therapy_tracks, ["discipline", "therapy_discipline"])
            start_col = pick_first_col(therapy_tracks, ["start_at", "started_at", "start_date", "created_at"])
            end_col = pick_first_col(therapy_tracks, ["end_at", "ended_at", "end_date"])
            if discipline_col and start_col:
                tt = therapy_tracks[["resident_id", discipline_col, start_col] + ([end_col] if end_col else [])].copy()
                tt["discipline"] = tt[discipline_col].astype(str).str.upper()
                tt["start_time"] = pd.to_datetime(tt[start_col], errors="coerce")
                if end_col:
                    tt["end_time"] = pd.to_datetime(tt[end_col], errors="coerce")
                else:
                    tt["end_time"] = pd.NaT
                tt = tt[tt["resident_id"].notna() & tt["start_time"].notna()]
                tt = tt[tt["discipline"] == "PT"]
                tt_by_resident = {
                    resident_id: group
                    for resident_id, group in tt.groupby("resident_id", sort=False)
                }

                for resident_id, snap_group in work.groupby("resident_id", sort=False):
                    tt_group = tt_by_resident.get(resident_id)
                    if tt_group is None or tt_group.empty:
                        continue
                    snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
                    start_times = tt_group["start_time"].to_numpy(dtype="datetime64[ns]")
                    end_times = tt_group["end_time"].to_numpy(dtype="datetime64[ns]")
                    end_na = pd.isna(end_times)

                    active_vals = np.zeros(len(snap_times), dtype=int)
                    done_vals = np.zeros(len(snap_times), dtype=int)
                    for i, snap_time in enumerate(snap_times):
                        active = (start_times <= snap_time) & (end_na | (end_times > snap_time))
                        done = (~end_na) & (end_times < snap_time)
                        active_vals[i] = int(active.any())
                        done_vals[i] = int(done.any())

                    work.loc[snap_group.index, "pt_active_flag"] = active_vals
                    work.loc[snap_group.index, "pt_done_before_flag"] = done_vals

    return work
