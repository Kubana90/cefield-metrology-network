"""
Unit tests for CEFIELD Baseline Engine, Prediction Engine, and Normalizer.

All tests use mocked SQLAlchemy sessions — no database required.
Run: pytest tests/ -v
"""

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

# conftest.py adds cloud-core to sys.path — imports work without qualification
from baseline import (
    CRITICAL_Q_FACTOR_DEFAULT,
    Z_SCORE_ALERT_THRESHOLD,
    compute_node_baseline,
    predict_time_to_failure,
)
from normalizer import HARDWARE_PROFILES, list_supported_hardware, normalize_vector


# ─── Shared Helpers ──────────────────────────────────────────────────────────────────────
def _make_meas(n: int, start_q: float, slope_per_sample: float = 0.0):
    """Generate n mock measurement objects (newest first, descending in time).

    slope_per_sample is applied in chronological order: start_q is the oldest
    value and each subsequent sample (going forward in time) shifts by
    slope_per_sample.  Because the list is newest-first, index 0 holds the
    most recent measurement: start_q + slope_per_sample * (n - 1).
    """
    now = datetime.now(UTC)
    return [
        MagicMock(
            q_factor=start_q + slope_per_sample * (n - 1 - i),
            timestamp=now - timedelta(hours=i),
        )
        for i in range(n)
    ]


def _mock_db(measurements):
    """Create a mock SQLAlchemy Session that returns given measurements."""
    db = MagicMock()
    (
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value
    ) = measurements
    return db


# ─── compute_node_baseline ──────────────────────────────────────────────────────────────────
class TestComputeNodeBaseline:
    def test_insufficient_data_returns_warmup(self):
        result = compute_node_baseline(_mock_db(_make_meas(5, 50_000)), node_id=1)
        assert result["needs_warmup"] is True
        assert result["n_samples"] == 5
        assert result["status"] == "insufficient_data"

    def test_stable_device_ok_status(self):
        result = compute_node_baseline(_mock_db(_make_meas(50, 50_000)), node_id=1)
        assert result["status"] == "ok"
        assert result["n_samples"] == 50
        assert not result["needs_warmup"]

    def test_stable_device_no_anomaly(self):
        result = compute_node_baseline(_mock_db(_make_meas(50, 50_000)), node_id=1)
        assert result["is_statistical_anomaly"] is False
        assert abs(result["z_score_last"]) < 1.0

    def test_statistical_anomaly_detected_on_sudden_drop(self):
        meas = _make_meas(50, 50_000)  # baseline ~50k
        meas[0].q_factor = 3_000  # severe drop on latest measurement
        result = compute_node_baseline(_mock_db(meas), node_id=1)
        assert result["is_statistical_anomaly"] is True
        assert result["z_score_last"] < -Z_SCORE_ALERT_THRESHOLD

    def test_mean_is_approximately_correct(self):
        result = compute_node_baseline(_mock_db(_make_meas(50, 40_000)), node_id=1)
        assert abs(result["mean_q"] - 40_000) < 1_000

    def test_negative_slope_for_degrading_device(self):
        # slope=-500/hour = very clear degradation
        result = compute_node_baseline(
            _mock_db(_make_meas(50, 50_000, slope_per_sample=-500)), node_id=1
        )
        assert result["slope_per_day"] < 0

    def test_positive_slope_for_improving_device(self):
        result = compute_node_baseline(
            _mock_db(_make_meas(30, 10_000, slope_per_sample=100)), node_id=1
        )
        assert result["slope_per_day"] > 0


# ─── predict_time_to_failure ────────────────────────────────────────────────────────────────
class TestPredictTimeToFailure:
    def test_stable_device_returns_none(self):
        baseline = {
            "status": "ok",
            "mean_q": 50_000,
            "slope_per_day": 10.0,
            "n_samples": 50,
            "needs_warmup": False,
        }
        assert predict_time_to_failure(baseline) is None

    def test_warmup_baseline_returns_none(self):
        baseline = {"status": "insufficient_data", "needs_warmup": True, "n_samples": 5}
        assert predict_time_to_failure(baseline) is None

    def test_degrading_device_predicts_correct_days(self):
        # mean=15000, slope=-200/day, threshold=5000 → (15000-5000)/200 = 50 days
        baseline = {
            "status": "ok",
            "mean_q": 15_000.0,
            "slope_per_day": -200.0,
            "n_samples": 60,
            "needs_warmup": False,
        }
        result = predict_time_to_failure(baseline, critical_threshold=5_000)
        assert result is not None
        assert result["days_to_failure"] == pytest.approx(50.0, rel=0.05)
        assert result["status"] == "degrading"
        assert result["confidence"] == "high"

    def test_already_critical_returns_zero_days(self):
        baseline = {
            "status": "ok",
            "mean_q": 2_000.0,
            "slope_per_day": -100.0,
            "n_samples": 30,
            "needs_warmup": False,
        }
        result = predict_time_to_failure(baseline, critical_threshold=5_000)
        assert result["days_to_failure"] == 0
        assert result["status"] == "already_critical"

    def test_confidence_scales_with_sample_count(self):
        base = {
            "status": "ok",
            "mean_q": 20_000.0,
            "slope_per_day": -100.0,
            "needs_warmup": False,
        }
        assert predict_time_to_failure({**base, "n_samples": 60})["confidence"] == "high"
        assert predict_time_to_failure({**base, "n_samples": 25})["confidence"] == "medium"
        assert predict_time_to_failure({**base, "n_samples": 12})["confidence"] == "low"


# ─── normalize_vector ───────────────────────────────────────────────────────────────────────
class TestNormalizeVector:
    def test_output_length_is_128(self):
        result = normalize_vector([0.5] * 128, "red_pitaya")
        assert len(result) == 128

    def test_output_is_unit_l2_vector(self):
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(128).tolist()
        result = normalize_vector(vec, "keysight_dso")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6, f"L2 norm = {norm}, expected 1.0"

    def test_different_hardware_produces_different_output(self):
        rng = np.random.default_rng(7)
        vec = rng.standard_normal(128).tolist()
        v_rp = normalize_vector(vec, "red_pitaya")
        v_ks = normalize_vector(vec, "keysight_dso")
        assert not np.allclose(v_rp, v_ks), (
            "Different hardware should yield different normalization"
        )

    def test_unknown_hardware_falls_back_gracefully(self):
        vec = [0.1] * 128
        result = normalize_vector(vec, "unknown_mystery_instrument_v9000")
        assert len(result) == 128
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6

    def test_all_supported_hardware_produce_valid_output(self):
        """Every hardware profile in HARDWARE_PROFILES must produce valid output."""
        rng = np.random.default_rng(99)
        vec = rng.standard_normal(128).tolist()
        for hw in list_supported_hardware():
            result = normalize_vector(vec, hw)
            assert len(result) == 128, f"Length error for {hw}"
            norm = np.linalg.norm(result)
            assert abs(norm - 1.0) < 1e-5, f"L2 norm error for {hw}: {norm}"

    def test_zero_vector_does_not_raise(self):
        """Edge case: all-zero vector must not raise ZeroDivisionError."""
        result = normalize_vector([0.0] * 128, "red_pitaya")
        assert len(result) == 128
