"""
Unit tests for src/data/build_dataset.py.
Run with:  pytest tests/test_build_dataset.py -v
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data.build_dataset import add_datetime, derive_base_columns, validate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n_days=2, periods=48):
    """Create a minimal valid raw DataFrame."""
    rows = []
    for d in range(n_days):
        day = date(2025, 5, 1 + d).isoformat()
        for p in range(1, periods + 1):
            rows.append({
                "settlement_date": day,
                "settlement_period": p,
                "ssp": 80.0 + p,
                "sbp": 82.0 + p,
                "net_imbalance_volume": -100.0 + p,
                "sell_price_adjustment": 0.0,
                "buy_price_adjustment": 0.0,
                "price_derivation_code": "N",
                "replacement_price": None,
            })
    df = pd.DataFrame(rows)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"]).dt.date
    return df


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestValidate:
    def test_clean_data_passes(self):
        df = make_df(n_days=2, periods=48)
        result = validate(df)
        assert len(result) == 2 * 48

    def test_drops_duplicate_rows(self):
        df = make_df(n_days=1, periods=48)
        df_duped = pd.concat([df, df.iloc[:5]], ignore_index=True)
        result = validate(df_duped)
        assert len(result) == 48

    def test_warns_on_missing_periods(self, caplog):
        df = make_df(n_days=2, periods=48)
        df_missing = df.iloc[5:].reset_index(drop=True)  # drop first 5 rows
        with caplog.at_level("WARNING"):
            validate(df_missing)
        assert "missing" in caplog.text.lower()


# ---------------------------------------------------------------------------
# add_datetime()
# ---------------------------------------------------------------------------

class TestAddDatetime:
    def test_column_created(self):
        df = make_df(n_days=1, periods=48)
        result = add_datetime(df)
        assert "settlement_datetime" in result.columns

    def test_period_1_is_midnight(self):
        df = make_df(n_days=1, periods=1)
        result = add_datetime(df)
        dt = result.loc[0, "settlement_datetime"]
        assert dt.hour == 0 and dt.minute == 0

    def test_period_2_is_half_past_midnight(self):
        df = make_df(n_days=1, periods=2)
        result = add_datetime(df)
        dt = result.loc[result["settlement_period"] == 2, "settlement_datetime"].iloc[0]
        assert dt.hour == 0 and dt.minute == 30

    def test_period_48_is_2330(self):
        df = make_df(n_days=1, periods=48)
        result = add_datetime(df)
        dt = result.loc[result["settlement_period"] == 48, "settlement_datetime"].iloc[0]
        assert dt.hour == 23 and dt.minute == 30

    def test_no_nulls(self):
        df = make_df(n_days=3, periods=48)
        result = add_datetime(df)
        assert result["settlement_datetime"].isnull().sum() == 0


# ---------------------------------------------------------------------------
# derive_base_columns()
# ---------------------------------------------------------------------------

class TestDeriveBaseColumns:
    def test_price_derivation_code_P_is_binary(self):
        df = make_df(n_days=1, periods=4)
        df.loc[df["settlement_period"] <= 2, "price_derivation_code"] = "P"
        result = derive_base_columns(df)
        assert set(result["price_derivation_code_P"].unique()).issubset({0, 1})
        assert result.loc[result["settlement_period"] <= 2, "price_derivation_code_P"].eq(1).all()
        assert result.loc[result["settlement_period"] > 2, "price_derivation_code_P"].eq(0).all()

    def test_abs_imbalance_is_positive(self):
        df = make_df(n_days=1, periods=48)
        result = derive_base_columns(df)
        assert (result["abs_imbalance_volume"] >= 0).all()

    def test_abs_imbalance_matches_niv(self):
        df = make_df(n_days=1, periods=48)
        result = derive_base_columns(df)
        expected = result["net_imbalance_volume"].abs()
        pd.testing.assert_series_equal(result["abs_imbalance_volume"], expected, check_names=False)
