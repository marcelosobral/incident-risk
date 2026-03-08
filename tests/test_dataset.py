from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


class BaseDatavisionMixin:
    dataset_filename: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        if not cls.dataset_filename:
            raise ValueError("dataset_filename must be set on the subclass")
        root = Path(__file__).resolve().parents[1]
        cls.dataset_path = root / "outputs" / cls.dataset_filename
        if not cls.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {cls.dataset_path}. Run build_dataset first."
            )
        cls.df = pd.read_parquet(cls.dataset_path)

    def test_required_columns_present(self) -> None:
        required = {
            "resident_id",
            "facility_id",
            "snapshot_date",
            "label_fall_30d",
            "label_rth_30d",
        }
        missing = required - set(self.df.columns)
        self.assertFalse(missing, f"Missing required columns: {missing}")

    def test_snapshot_date_range(self) -> None:
        dates = pd.to_datetime(self.df["snapshot_date"], errors="coerce")
        self.assertFalse(dates.isna().any(), "snapshot_date has NaT values")
        # Weekly snapshots on Monday
        self.assertTrue((dates.dt.dayofweek == 0).all(), "snapshot_date not on Monday")


class TestDatavisionWeekly2025(BaseDatavisionMixin, unittest.TestCase):
    dataset_filename = "datavision_weekly_2025.parquet"

    def test_snapshot_date_range(self) -> None:
        dates = pd.to_datetime(self.df["snapshot_date"], errors="coerce")
        self.assertFalse(dates.isna().any(), "snapshot_date has NaT values")
        self.assertTrue((dates.dt.year == 2025).all(), "snapshot_date not in 2025")
        self.assertTrue((dates.dt.dayofweek == 0).all(), "snapshot_date not on Monday")


class TestDatavisionWeekly2023to2025(BaseDatavisionMixin, unittest.TestCase):
    dataset_filename = "datavision_weekly_2023-08_2025-01.parquet"

    def test_snapshot_date_range(self) -> None:
        dates = pd.to_datetime(self.df["snapshot_date"], errors="coerce")
        self.assertFalse(dates.isna().any(), "snapshot_date has NaT values")
        self.assertTrue((dates >= pd.Timestamp("2023-08-01")).all())
        self.assertTrue((dates <= pd.Timestamp("2025-01-31")).all())
        self.assertTrue((dates.dt.dayofweek == 0).all(), "snapshot_date not on Monday")

    def test_labels_binary(self) -> None:
        for col in ("label_fall_30d", "label_rth_30d"):
            series = self.df[col].dropna()
            invalid = series[~series.isin([0, 1])]
            self.assertTrue(invalid.empty, f"{col} has non-binary values")

    def test_counts_nonnegative(self) -> None:
        count_cols = [c for c in self.df.columns if c.endswith("_count_30d") or c.endswith("_count_90d") or c.endswith("_count_180d")]
        if not count_cols:
            self.skipTest("No count columns found")
        for col in count_cols:
            self.assertTrue((self.df[col] >= 0).all(), f"{col} has negative values")

    def test_days_since_last_nonnegative(self) -> None:
        recency_cols = [c for c in self.df.columns if c.endswith("_days_since_last")]
        if not recency_cols:
            self.skipTest("No recency columns found")
        for col in recency_cols:
            series = self.df[col].dropna()
            self.assertTrue((series >= 0).all(), f"{col} has negative values")

    def test_unique_snapshot_rows(self) -> None:
        key_cols = ["resident_id", "facility_id", "snapshot_date"]
        duplicates = self.df.duplicated(subset=key_cols).sum()
        self.assertEqual(duplicates, 0, "Duplicate resident/facility/snapshot rows found")


if __name__ == "__main__":
    unittest.main()
