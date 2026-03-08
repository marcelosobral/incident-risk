from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.time_utils import filter_table_by_date_range


def load_parquet_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in data_dir.glob("*.parquet"):
        tables[path.stem] = pd.read_parquet(path)
    return tables


def filter_tables_by_date_range(
    tables: dict[str, pd.DataFrame],
    min_year: int,
    max_year: int,
    exclude_cols: Iterable[str],
) -> dict[str, pd.DataFrame]:
    filtered: dict[str, pd.DataFrame] = {}
    for name, df in tables.items():
        filtered[name] = filter_table_by_date_range(
            df,
            min_year=min_year,
            max_year=max_year,
            exclude_cols=exclude_cols,
        )
    return filtered
