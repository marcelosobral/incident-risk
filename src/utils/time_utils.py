from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

TIME_COL_HINTS = (
    "date",
    "time",
    "timestamp",
    "created",
    "updated",
    "start",
    "end",
    "occurred",
    "measured",
    "admit",
    "discharge",
)

PREFERRED_TIME_COLS: dict[str, list[str]] = {
    "adl_responses": ["assessment_date", "created_at"],
    "care_plans": ["initiated_at", "created_at"],
    "diagnoses": ["onset_at", "created_at"],
    "document_tags": ["created_at"],
    "factors": ["created_at"],
    "gg_responses": ["created_at"],
    "hospital_admissions": [
        "effective_date",
        "admission_date",
        "admit_date",
        "admitted_at",
        "admit_at",
        "start_at",
    ],
    "hospital_transfers": [
        "effective_date",
        "transfer_date",
        "transfer_at",
        "start_at",
    ],
    "incidents": ["occurred_at", "created_at"],
    "injuries": ["created_at"],
    "lab_reports": ["reported_at", "collected_at", "created_at"],
    "medications": ["administered_at", "scheduled_at", "created_at"],
    "physician_orders": ["start_at", "created_at"],
    "vitals": ["measured_at", "observed_at", "created_at"],
}


def detect_time_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        lc = str(c).lower()
        if any(h in lc for h in TIME_COL_HINTS):
            cols.append(c)
    return cols


def pick_first_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def pick_time_col_for_table(table_name: str, df: pd.DataFrame) -> Optional[str]:
    preferred = PREFERRED_TIME_COLS.get(table_name, [])
    col = pick_first_col(df, preferred)
    if col:
        return col
    time_cols = detect_time_cols(df)
    return time_cols[0] if time_cols else None


def filter_table_by_date_range(
    df: pd.DataFrame,
    min_year: int,
    max_year: int,
    exclude_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    exclude = {str(c).lower() for c in (exclude_cols or [])}
    keep = pd.Series(True, index=df.index)
    for col in detect_time_cols(df):
        if str(col).lower() in exclude:
            continue
        ts = pd.to_datetime(df[col], errors="coerce")
        in_range = (ts.dt.year >= min_year) & (ts.dt.year <= max_year)
        keep &= in_range | ts.isna()
    return df.loc[keep].copy()
