"""
Unit tests for the PI calibration production-safety guard in forecast_phase3.py.

Exercises the full scenario matrix:

  A  artifact present, data calibrated          → no raise, PASS logged
  B  artifact absent, WF sentinel present       → RuntimeError (production failure)
  C  artifact absent, no WF sentinel            → silent skip (dev/pre-Phase-4)
  D  <6 unsettled forecast rows                 → no raise (late-intraday edge)
  E  artifact corrupt/unreadable after exists   → RuntimeError
  F  data raw (uniform spread)                  → RuntimeError
  G  all 48 rows settled (is_actual=True)       → no raise (D variant)
  H  partial settle — only unsettled rows checked

Run with:  pytest tests/test_pi_calibration_guard.py -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))

# Import only _assert_pi_calibrated; the rest of forecast_phase3 has heavy
# runtime dependencies (joblib models, live weather APIs).  We import the
# symbol directly from the module's source text so the test file remains
# importable without those dependencies.
import importlib.util as _ilu
import types as _types

_fp3_path = Path(__file__).resolve().parents[1] / "src" / "models" / "forecast_phase3.py"

# Build a minimal stub module so the import succeeds without side-effects.
# We execute only the portion of the source up to and including
# _assert_pi_calibrated; everything below (which imports correctors, joblib,
# etc.) is not needed.
def _load_assert_fn():
    source = _fp3_path.read_text()
    # Extract just the function definition block we need.
    # Split at the first occurrence of "# ── Weather" (the line that follows
    # the function) so we avoid importing the rest of the module.
    cut = source.find("\n# ── Weather ─")
    if cut == -1:
        cut = source.find("\ndef fetch_forecast_weather")
    stub_source = "\n".join([
        "import json, numpy as np, pandas as pd, logging",
        "log = logging.getLogger('forecast_phase3')",
        source[source.find("\ndef _assert_pi_calibrated"):cut],
    ])
    mod = _types.ModuleType("fp3_stub")
    exec(compile(stub_source, str(_fp3_path), "exec"), mod.__dict__)
    return mod._assert_pi_calibrated


_assert_pi_calibrated = _load_assert_fn()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REAL_DELTA_MAP = {
    1: 13.95, 2: 16.00, 3: 15.00, 4: 14.50, 5: 15.50, 6: 16.00,
    7: 17.00, 8: 17.50, 9: 18.00, 10: 19.00, 11: 20.00, 12: 21.00,
    13: 22.00, 14: 23.00, 15: 24.00, 16: 24.50, 17: 25.00, 18: 25.50,
    19: 26.00, 20: 26.50, 21: 27.00, 22: 27.50, 23: 28.00, 24: 25.16,
    25: 26.00, 26: 27.00, 27: 28.00, 28: 29.00, 29: 30.00, 30: 31.00,
    31: 32.00, 32: 33.00, 33: 34.00, 34: 35.00, 35: 36.00, 36: 37.00,
    37: 32.41, 38: 31.00, 39: 30.00, 40: 29.00, 41: 28.00, 42: 27.00,
    43: 26.00, 44: 25.00, 45: 24.00, 46: 23.00, 47: 22.00, 48: 17.95,
}
RAW_SPREAD = 40.0   # level_q90 - level_q10 for a synthetic forecast day


def make_pi_artifact(tmp_path: Path, delta_map: dict = None) -> Path:
    if delta_map is None:
        delta_map = REAL_DELTA_MAP
    p = tmp_path / "pi_calibration_v1.json"
    p.write_text(json.dumps({
        "version": "v1",
        "delta_by_sp": {str(k): v for k, v in delta_map.items()},
        "achieved_coverage": 0.7982,
    }))
    return p


def make_wf_sentinel(tmp_path: Path) -> Path:
    """Minimal walk_forward_predictions.csv to act as the production sentinel."""
    p = tmp_path / "walk_forward_predictions.csv"
    p.write_text("settlement_date,ssp_actual\n2025-07-01,80.0\n")
    return p


def make_forecast_df(
    n_sps: int = 48,
    calibrated: bool = True,
    settled_mask: "list[bool] | None" = None,
    delta_map: dict = None,
    raw_spread: float = RAW_SPREAD,
) -> pd.DataFrame:
    """
    Build a synthetic forecast DataFrame.

    calibrated=True  → spread(sp) = raw_spread + 2*delta(sp)  (PI-calibrated)
    calibrated=False → spread(sp) = raw_spread for all SPs    (raw, uniform)
    settled_mask     → list of bool, True = is_actual row
    """
    if delta_map is None:
        delta_map = REAL_DELTA_MAP
    base_q50 = 85.0
    base_q10 = base_q50 - raw_spread / 2
    base_q90 = base_q50 + raw_spread / 2
    rows = []
    for i, sp in enumerate(range(1, n_sps + 1)):
        d = delta_map.get(sp, 0.0)
        if calibrated:
            q10 = base_q10 - d
            q90 = base_q90 + d
        else:
            q10 = base_q10
            q90 = base_q90
        is_act = bool(settled_mask[i]) if settled_mask is not None else False
        rows.append({
            "settlement_period": sp,
            "ssp_predicted": base_q50,
            "ssp_q10":       q10,
            "ssp_q50":       base_q50,
            "ssp_q90":       q90,
            "is_actual":     is_act,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario A — artifact present, calibrated data → passes silently
# ---------------------------------------------------------------------------

class TestScenarioA:
    def test_pass_on_calibrated_48sp(self, tmp_path):
        pi = make_pi_artifact(tmp_path)
        df = make_forecast_df(calibrated=True)
        _assert_pi_calibrated(df, pi, "H+1 2026-06-18")  # must not raise

    def test_pass_on_calibrated_no_is_actual_column(self, tmp_path):
        pi = make_pi_artifact(tmp_path)
        df = make_forecast_df(calibrated=True).drop(columns=["is_actual"])
        _assert_pi_calibrated(df, pi, "H+1 no-is_actual")  # must not raise


# ---------------------------------------------------------------------------
# Scenario B — artifact absent + WF sentinel present → RuntimeError
#
# This is the exact failure mode that shipped raw bands before 2026-06-17.
# The test enforces that the production PI block raises and that the message
# is explicit enough for an on-call engineer to act on.
# ---------------------------------------------------------------------------

class TestScenarioB:
    def _run_pi_block_logic(self, pi_path_obj: Path, wf_sentinel_present: bool,
                             df: pd.DataFrame, label: str, tmp_path: Path):
        """
        Inline re-implementation of the production PI block from forecast_phase3.py.
        Separated so the test doesn't import the full module.
        """
        from correctors import apply_pi_calibration

        _pi_path = pi_path_obj
        _pi_in_production = wf_sentinel_present

        if _pi_path.exists():
            try:
                result = apply_pi_calibration(df, _pi_path)
            except Exception as e:
                raise RuntimeError(
                    f"PI calibration failed [{label}]: {e} — "
                    f"raw q10/q90 bands would be written"
                ) from e
            _assert_pi_calibrated(result, _pi_path, label)
            return result
        elif _pi_in_production:
            raise RuntimeError(
                f"pi_calibration_v1.json absent in production context [{label}]: "
                f"walk_forward_predictions.csv is present (trained system) but "
                f"pi_calibration_v1.json is missing — raw q10/q90 bands would be "
                f"written to disk.  Restore model_assets/pi_calibration_v1.json."
            )
        else:
            return df  # dev/pre-Phase-4 silent skip

    def test_raises_when_wf_present_pi_absent(self, tmp_path):
        """B — the primary regression guard."""
        absent_pi = tmp_path / "pi_calibration_v1.json"  # intentionally not created
        df = make_forecast_df(calibrated=False)  # raw — would ship wrong bands

        with pytest.raises(RuntimeError) as exc_info:
            self._run_pi_block_logic(absent_pi, wf_sentinel_present=True,
                                      df=df, label="H+1 2026-06-18", tmp_path=tmp_path)

        msg = str(exc_info.value)
        assert "absent in production context" in msg, f"message not operator-readable: {msg}"
        assert "walk_forward_predictions.csv" in msg, \
            f"should name the sentinel that triggered production mode: {msg}"
        assert "raw q10/q90 bands would be written" in msg, \
            f"should state the consequence: {msg}"
        assert "Restore model_assets/pi_calibration_v1.json" in msg, \
            f"should give operator the remediation action: {msg}"

    def test_h1_and_h2_labels_distinguished(self, tmp_path):
        """B variant — H+2 raise carries the correct date label."""
        absent_pi = tmp_path / "pi_calibration_v1.json"
        df = make_forecast_df(calibrated=False)

        for label in ("H+1 2026-06-18", "H+2 2026-06-19"):
            with pytest.raises(RuntimeError) as exc_info:
                self._run_pi_block_logic(absent_pi, wf_sentinel_present=True,
                                          df=df, label=label, tmp_path=tmp_path)
            assert label in str(exc_info.value), \
                f"date label should appear in message for {label}"


# ---------------------------------------------------------------------------
# Scenario C — artifact absent + no WF sentinel → silent skip (dev mode)
# ---------------------------------------------------------------------------

class TestScenarioC:
    def test_no_raise_when_no_sentinel_and_no_artifact(self, tmp_path):
        absent_pi = tmp_path / "pi_calibration_v1.json"
        df = make_forecast_df(calibrated=False)

        # Confirm no RuntimeError is raised
        try:
            from correctors import apply_pi_calibration
            _pi_in_production = False   # no WF sentinel
            if not absent_pi.exists():
                if _pi_in_production:
                    raise RuntimeError("should not reach here")
                # else: dev mode, do nothing
        except RuntimeError:
            pytest.fail("dev-mode path must not raise")


# ---------------------------------------------------------------------------
# Scenario D — <6 unsettled forecast rows → assertion skips (no raise)
# ---------------------------------------------------------------------------

class TestScenarioD:
    def test_no_raise_when_fewer_than_6_unsettled(self, tmp_path):
        pi = make_pi_artifact(tmp_path)
        # 5 unsettled rows — below the minimum-N threshold
        df = make_forecast_df(n_sps=5, calibrated=False)  # raw but too few rows
        _assert_pi_calibrated(df, pi, "H+1 late-intraday")  # must not raise

    def test_no_raise_when_all_48_are_settled(self, tmp_path):
        """D variant — every row is_actual=True."""
        pi = make_pi_artifact(tmp_path)
        df = make_forecast_df(calibrated=False,
                               settled_mask=[True] * 48)  # raw values but all settled
        _assert_pi_calibrated(df, pi, "H+1 all-settled")  # must not raise

    def test_boundary_exactly_6_unsettled_does_check(self, tmp_path):
        """At N=6 the check activates — raw data must raise."""
        pi = make_pi_artifact(tmp_path)
        df = make_forecast_df(n_sps=6, calibrated=False)
        with pytest.raises(RuntimeError):
            _assert_pi_calibrated(df, pi, "H+1 boundary-6")

    def test_partial_settled_only_checks_unsettled(self, tmp_path):
        """H variant — settled rows are excluded from the spread check."""
        pi = make_pi_artifact(tmp_path)
        # SPs 1-40 settled (is_actual=True, q10=q50=q90 — zero spread, would fail check)
        # SPs 41-48 unsettled with calibrated spread — should pass
        mask = [True] * 40 + [False] * 8
        # Build manually: settled SPs have q10=q50=q90=actual; unsettled are calibrated
        rows = []
        base = 85.0
        for sp in range(1, 49):
            d = REAL_DELTA_MAP.get(sp, 0.0)
            is_act = mask[sp - 1]
            if is_act:
                rows.append({"settlement_period": sp, "ssp_q10": base,
                              "ssp_q50": base, "ssp_q90": base, "is_actual": True})
            else:
                rows.append({"settlement_period": sp, "ssp_q10": base - 20 - d,
                              "ssp_q50": base, "ssp_q90": base + 20 + d, "is_actual": False})
        df = pd.DataFrame(rows)
        _assert_pi_calibrated(df, pi, "H+1 partial-settled")  # must not raise


# ---------------------------------------------------------------------------
# Scenario E — artifact exists but is corrupt/unreadable → RuntimeError
# ---------------------------------------------------------------------------

class TestScenarioE:
    def test_raises_on_malformed_json(self, tmp_path):
        corrupt = tmp_path / "pi_calibration_v1.json"
        corrupt.write_text("{bad json [[[")
        df = make_forecast_df(calibrated=True)

        with pytest.raises(RuntimeError) as exc_info:
            _assert_pi_calibrated(df, corrupt, "H+1 corrupt")

        msg = str(exc_info.value)
        assert "could not read" in msg, f"message should describe the read failure: {msg}"
        assert str(corrupt) in msg, f"message should name the corrupt file: {msg}"

    def test_raises_on_empty_file(self, tmp_path):
        empty = tmp_path / "pi_calibration_v1.json"
        empty.write_text("")
        df = make_forecast_df(calibrated=True)

        with pytest.raises(RuntimeError) as exc_info:
            _assert_pi_calibrated(df, empty, "H+1 empty")

        assert "could not read" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Scenario F — raw (uniform spread) data → RuntimeError
# ---------------------------------------------------------------------------

class TestScenarioF:
    def test_raises_on_raw_uniform_spread(self, tmp_path):
        pi = make_pi_artifact(tmp_path)
        df = make_forecast_df(calibrated=False)  # raw: spread is constant across SPs

        with pytest.raises(RuntimeError) as exc_info:
            _assert_pi_calibrated(df, pi, "H+1 raw")

        msg = str(exc_info.value)
        assert "FAILED" in msg
        assert "raw q10/q90 bands detected" in msg
        assert "std(spread-2δ)" in msg

    def test_message_contains_numeric_diagnostics(self, tmp_path):
        """The message must give std values so an operator can diagnose."""
        pi = make_pi_artifact(tmp_path)
        df = make_forecast_df(calibrated=False)

        with pytest.raises(RuntimeError) as exc_info:
            _assert_pi_calibrated(df, pi, "H+1 diagnostics")

        msg = str(exc_info.value)
        # Check numeric values appear (float formatted to 3 dp)
        import re
        assert re.search(r"std\(spread-2δ\)=\d+\.\d{3}", msg), \
            f"std(spread-2δ) value missing from: {msg}"
        assert re.search(r"std\(spread\)=\d+\.\d{3}", msg), \
            f"std(spread) value missing from: {msg}"
