from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.time_utils import pick_first_col


def label_from_events(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    horizon_days: int,
    label_name: str,
) -> pd.DataFrame:
    work = snapshots.copy()
    work[label_name] = 0
    if events.empty:
        return work

    events = events.sort_values(["resident_id", "event_time"])
    work = work.sort_values(["resident_id", "snapshot_date"]).reset_index(drop=True)

    for resident_id, snap_group in work.groupby("resident_id", sort=False):
        ev_group = events[events["resident_id"] == resident_id]
        if ev_group.empty:
            continue
        ev_times = ev_group["event_time"].to_numpy(dtype="datetime64[ns]")
        snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")
        start_idx = np.searchsorted(ev_times, snap_times, side="right")
        end_idx = np.searchsorted(
            ev_times,
            snap_times + np.timedelta64(horizon_days, "D"),
            side="right",
        )
        labels = (end_idx > start_idx).astype(int)
        work.loc[snap_group.index, label_name] = labels

    return work


def build_fall_events(incidents: pd.DataFrame) -> pd.DataFrame:
    if incidents.empty:
        return pd.DataFrame(columns=["resident_id", "event_time"])
    if "incident_type" not in incidents.columns:
        return pd.DataFrame(columns=["resident_id", "event_time"])

    work = incidents.copy()
    work["incident_type"] = work["incident_type"].astype(str)
    fall_mask = work["incident_type"].str.lower().str.contains("fall", na=False)
    work = work[fall_mask]
    if "occurred_at" not in work.columns:
        return pd.DataFrame(columns=["resident_id", "event_time"])

    work["event_time"] = pd.to_datetime(work["occurred_at"], errors="coerce")
    work = work[work["resident_id"].notna() & work["event_time"].notna()]
    return work[["resident_id", "event_time"]]


def build_rth_events(hospital_admissions: pd.DataFrame) -> pd.DataFrame:
    if hospital_admissions.empty:
        return pd.DataFrame(columns=["resident_id", "event_time"])

    ha = hospital_admissions.copy()

    admit_col = pick_first_col(
        ha,
        [
            "effective_date",
            "admission_date",
            "admit_date",
            "admitted_at",
            "admit_at",
            "start_at",
        ],
    )
    status_col = pick_first_col(
        ha,
        [
            "admission_status",
            "stay_purpose",
            "admission_type",
            "transfer_reason",
            "reason",
        ],
    )

    if admit_col is None or "resident_id" not in ha.columns:
        return pd.DataFrame(columns=["resident_id", "event_time"])

    ha["event_time"] = pd.to_datetime(ha[admit_col], errors="coerce")
    planned_mask = pd.Series(False, index=ha.index)
    if status_col:
        status = ha[status_col].astype(str).str.lower()
        planned_mask = status.str.contains("planned|elective|scheduled", na=False)
        ha = ha[~planned_mask]
    ha = ha[ha["resident_id"].notna() & ha["event_time"].notna()]
    return ha[["resident_id", "event_time"]]
