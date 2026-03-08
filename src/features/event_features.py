from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def make_event_table(
    df: pd.DataFrame,
    time_col: str,
    resident_col: str = "resident_id",
    facility_col: str = "facility_id",
) -> pd.DataFrame:
    if df.empty or time_col not in df.columns:
        return pd.DataFrame(columns=[resident_col, facility_col, "event_time"])
    events = df[[resident_col, facility_col, time_col]].copy()
    events["event_time"] = pd.to_datetime(events[time_col], errors="coerce")
    events = events[events[resident_col].notna() & events["event_time"].notna()]
    return events[[resident_col, facility_col, "event_time"]]


def build_event_features(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    windows_days: Iterable[int],
    prefix: str,
) -> pd.DataFrame:
    if events.empty:
        for w in windows_days:
            snapshots[f"{prefix}_count_{w}d"] = 0
        snapshots[f"{prefix}_days_since_last"] = np.nan
        return snapshots

    work = snapshots.copy()
    work = work.sort_values(["resident_id", "snapshot_date"]).reset_index(drop=True)

    for w in windows_days:
        work[f"{prefix}_count_{w}d"] = 0
    work[f"{prefix}_days_since_last"] = np.nan

    events = events.sort_values(["resident_id", "event_time"])

    for resident_id, snap_group in work.groupby("resident_id", sort=False):
        ev_group = events[events["resident_id"] == resident_id]
        if ev_group.empty:
            continue
        ev_times = ev_group["event_time"].to_numpy(dtype="datetime64[ns]")
        snap_times = snap_group["snapshot_date"].to_numpy(dtype="datetime64[ns]")

        right_idx = np.searchsorted(ev_times, snap_times, side="right")
        for w in windows_days:
            left_idx = np.searchsorted(
                ev_times,
                snap_times - np.timedelta64(w, "D"),
                side="right",
            )
            counts = right_idx - left_idx
            work.loc[snap_group.index, f"{prefix}_count_{w}d"] = counts

        recency = np.full(len(snap_times), np.nan)
        has_event = right_idx > 0
        last_idx = right_idx[has_event] - 1
        last_times = ev_times[last_idx]
        recency[has_event] = (snap_times[has_event] - last_times).astype("timedelta64[D]").astype(float)
        work.loc[snap_group.index, f"{prefix}_days_since_last"] = recency

    return work
