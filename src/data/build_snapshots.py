from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.time_utils import pick_first_col


def build_snapshot_calendar(
    start_date: str | datetime,
    end_date: str | datetime,
    freq: str = "W-MON",
) -> pd.DatetimeIndex:
    return pd.date_range(start_date, end_date, freq=freq)


def build_resident_snapshots(
    residents: pd.DataFrame,
    snapshot_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    if residents.empty:
        raise ValueError("residents table is empty.")

    admit_col = pick_first_col(
        residents,
        ["admission_date", "admit_date", "admitted_at", "admit_at", "start_at"],
    )
    discharge_col = pick_first_col(
        residents,
        ["discharge_date", "discharged_at", "discharge_at", "end_at", "end_date"],
    )
    if admit_col is None or discharge_col is None:
        raise ValueError("Missing admission or discharge columns in residents.")

    work = residents.copy()
    work["admit_ts"] = pd.to_datetime(work[admit_col], errors="coerce")
    work["discharge_ts"] = pd.to_datetime(work[discharge_col], errors="coerce")

    rows = []
    start_date = snapshot_dates.min()
    end_date = snapshot_dates.max()

    for _, row in work.iterrows():
        resident_id = row.get("resident_id")
        if pd.isna(resident_id):
            continue
        admit_ts = row.get("admit_ts")
        discharge_ts = row.get("discharge_ts")
        if pd.isna(admit_ts):
            continue
        if pd.isna(discharge_ts):
            discharge_ts = end_date

        window_start = max(admit_ts, start_date)
        window_end = min(discharge_ts, end_date)
        if window_start > window_end:
            continue

        active_dates = snapshot_dates[(snapshot_dates >= window_start) & (snapshot_dates <= window_end)]
        if active_dates.empty:
            continue
        for snap_date in active_dates:
            rows.append(
                {
                    "resident_id": resident_id,
                    "facility_id": row.get("facility_id"),
                    "snapshot_date": snap_date,
                }
            )

    snapshots = pd.DataFrame(rows)
    if snapshots.empty:
        raise ValueError("No snapshot rows generated. Check resident dates.")
    return snapshots
