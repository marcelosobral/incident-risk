from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.build_snapshots import build_resident_snapshots, build_snapshot_calendar
from src.data.load_raw import filter_tables_by_date_range, load_parquet_tables
from src.features.event_features import build_event_features, make_event_table
from src.features.derived_features import (
    add_care_plan_signals,
    add_comorbidity_flags,
    add_facility_effects,
    add_fall_injury_features,
    add_functional_status_trends,
    add_medication_risk,
    add_resident_risk_signals,
    add_resident_demographics,
    add_vital_type_features,
)
from src.features.labels import build_fall_events, build_rth_events, label_from_events
from src.utils.time_utils import pick_time_col_for_table

DATE_MIN_YEAR = 2000
DATE_MAX_YEAR = 2026
EXCLUDE_DATE_COLS = {"date_of_birth", "dob", "birth_date", "birthdate"}
FEATURE_WINDOWS_DAYS = [30, 90, 180]

FEATURE_TABLES = [
    "adl_responses",
    "care_plans",
    "diagnoses",
    "document_tags",
    "factors",
    "gg_responses",
    "hospital_admissions",
    "hospital_transfers",
    "incidents",
    "injuries",
    "lab_reports",
    "medications",
    "physician_orders",
    "therapy_tracks",
    "vitals",
]


def build_dataset(
    data_dir: Path,
    output_path: Path,
    date_begin: str = "2025-01-01",
    date_end: str = "2025-12-31",
    snapshot_freq: str = "W-MON",
) -> pd.DataFrame:
    tables = load_parquet_tables(data_dir)
    filtered = filter_tables_by_date_range(
        tables,
        min_year=DATE_MIN_YEAR,
        max_year=DATE_MAX_YEAR,
        exclude_cols=EXCLUDE_DATE_COLS,
    )

    residents = filtered.get("residents")
    if residents is None:
        raise ValueError("residents table is missing.")

    snapshot_dates = build_snapshot_calendar(date_begin, date_end, freq=snapshot_freq)
    snapshots = build_resident_snapshots(residents, snapshot_dates)

    for table_name in FEATURE_TABLES:
        df = filtered.get(table_name)
        if df is None or df.empty:
            continue
        if "resident_id" not in df.columns:
            continue
        time_col = pick_time_col_for_table(table_name, df)
        if time_col is None:
            continue
        events = make_event_table(df, time_col)
        snapshots = build_event_features(
            snapshots,
            events,
            windows_days=FEATURE_WINDOWS_DAYS,
            prefix=table_name,
        )

    snapshots = add_resident_demographics(snapshots, residents, copy=False)
    snapshots = add_comorbidity_flags(snapshots, filtered.get("diagnoses", pd.DataFrame()), copy=False)
    snapshots = add_functional_status_trends(
        snapshots,
        filtered.get("adl_responses", pd.DataFrame()),
        filtered.get("gg_responses", pd.DataFrame()),
        copy=False,
    )
    snapshots = add_vital_type_features(snapshots, filtered.get("vitals", pd.DataFrame()), copy=False)
    snapshots = add_medication_risk(snapshots, filtered.get("medications", pd.DataFrame()), copy=False)
    snapshots = add_fall_injury_features(
        snapshots,
        filtered.get("incidents", pd.DataFrame()),
        filtered.get("injuries", pd.DataFrame()),
        copy=False,
    )
    snapshots = add_facility_effects(
        snapshots,
        filtered.get("incidents", pd.DataFrame()),
        filtered.get("hospital_admissions", pd.DataFrame()),
        copy=False,
    )

    snapshots = add_resident_risk_signals(
        snapshots,
        filtered.get("gg_responses", pd.DataFrame()),
        filtered.get("medications", pd.DataFrame()),
        filtered.get("adl_responses", pd.DataFrame()),
        filtered.get("diagnoses", pd.DataFrame()),
        filtered.get("vitals", pd.DataFrame()),
        filtered.get("hospital_admissions", pd.DataFrame()),
        filtered.get("incidents", pd.DataFrame()),
        filtered.get("therapy_tracks", pd.DataFrame()),
        copy=False,
    )

    snapshots = add_care_plan_signals(snapshots, filtered.get("care_plans", pd.DataFrame()), copy=False)
    care_plan_drop = [
        c
        for c in snapshots.columns
        if c.startswith("care_plans_")
        and c not in {"care_plans_days_since_last", "care_plans_count_total"}
    ]
    if care_plan_drop:
        snapshots = snapshots.drop(columns=care_plan_drop)

    fall_events = build_fall_events(filtered.get("incidents", pd.DataFrame()))
    snapshots = label_from_events(
        snapshots,
        fall_events,
        horizon_days=30,
        label_name="label_fall_30d",
    )

    rth_events = build_rth_events(filtered.get("hospital_admissions", pd.DataFrame()))
    snapshots = label_from_events(
        snapshots,
        rth_events,
        horizon_days=30,
        label_name="label_rth_30d",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_parquet(output_path, index=False)
    return snapshots


def main() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    output_path = Path(__file__).resolve().parents[2] / "outputs" / "datavision_weekly_2025.parquet"
    build_dataset(data_dir, output_path)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
