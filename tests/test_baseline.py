"""
Unit tests for CEFIELD Baseline Engine and Normalizer.
Run: pytest tests/ -v
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock


# ─── Helpers ─────────────────────────────────────────────────────────────────────
def _make_measurements(n: int, start_q: float, slope_per_sample: float = 0.0):
    """Returns n mock measurement objects descending in time (newest first)."""
    now = datetime.now(timezone.utc)
    meas = []
    for i in range(n):
        m = MagicMock()
        m.q_factor = start_q + slope_per_sample * i
        m.timestamp = now - timedelta(hours=i)
        meas.append(m)
    return meas


def _db_returning(measurements):
    db = MagicMock()
    (
        db.query.return_value
        .filter.return_value
        .order_by.return_value
        .limit.return_value
        .all.return_value
    ) = measurements
    return db


# ─── Baseline Tests ───────────────────────────────────────────────────────────────
class TestComputeNodeBaseline:
    def test_insufficient_data_returns_warmup(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from baseline import compute_node_baseline
        db = _db_returning(_make_measurements(5, 50_000))
        result = compute_node_baseline(db, node_id=1)
        assert result["needs_warmup"] is True
        assert result["n_samples"] == 5

    def test_stable_device_no_anomaly(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from baseline import compute_node_baseline
        db = _db_returning(_make_measurements(50, 50_000, slope_per_sample=0))
        result = compute_node_baseline(db, node_id=1)
        assert result["status"] == "ok"
        assert result["is_statistical_anomaly"] is False
        assert abs(result["z_score_last"]) < 1.0

    def test_statistical_anomaly_detected(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from baseline import compute_node_baseline
        meas = _make_measurements(50, 50_000, slope_per_sample=0)
        meas[0].q_factor = 5_000  # Severe outlier vs baseline of 50k
        db = _db_returning(meas)
        result = compute_node_baseline(db, node_id=1)
        assert result["is_statistical_anomaly"] is True
        assert result["z_score_last"] < -2.5

    def test_degrading_slope_detected(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from baseline import compute_node_baseline
        # slope_per_sample=-200 over 50 samples = clear downward trend
        db = _db_returning(_make_measurements(50, 50_000, slope_per_sample=-200))
        result = compute_node_baseline(db, node_id=1)
        assert result["status"] == "ok"
        assert result["slope_per_day"] < 0


# ─── Prediction Tests ─────────────────────────────────────────────────────────────
class TestPredictTimeToFailure:
    def test_stable_returns_none(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from baseline import predict_time_to_failure
        baseline = {"status": "ok", "mean_q": 50_000, "slope_per_day": 10, "n_samples": 50}
        assert predict_time_to_failure(baseline) is None

    def test_warmup_returns_none(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from baseline import predict_time_to_failure
        baseline = {"status": "insufficient_data", "needs_warmup": True, "n_samples": 5}
        assert predict_time_to_failure(baseline) is None

    def test_degrading_predicts_correct_days(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from baseline import predict_time_to_failure
        # mean=15000, slope=-200/day, threshold=5000 → 50 days
        baseline = {
            "status": "ok", "mean_q": 15_000,
            "slope_per_day": -200.0, "n_samples": 60,
            "needs_warmup": False,
        }
        result = predict_time_to_failure(baseline, critical_threshold=5_000)
        assert result is not None
        assert result["days_to_failure"] == pytest.approx(50.0, rel=0.05)
        assert result["confidence"] == "high"
        assert result["status"] == "degrading"

    def test_already_critical_returns_zero(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from baseline import predict_time_to_failure
        baseline = {
            "status": "ok", "mean_q": 3_000,
            "slope_per_day": -100.0, "n_samples": 30,
            "needs_warmup": False,
        }
        result = predict_time_to_failure(baseline, critical_threshold=5_000)
        assert result["days_to_failure"] == 0
        assert result["status"] == "already_critical"


# ─── Normalizer Tests ─────────────────────────────────────────────────────────────
class TestNormalizer:
    def test_output_length_preserved(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from normalizer import normalize_vector
        vec = [0.5] * 128
        result = normalize_vector(vec, "red_pitaya")
        assert len(result) == 128

    def test_output_is_unit_vector(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from normalizer import normalize_vector
        vec = list(np.random.default_rng(42).standard_normal(128))
        result = normalize_vector(vec, "keysight_dso")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-6

    def test_different_hardware_different_output(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from normalizer import normalize_vector
        vec = list(np.random.default_rng(7).standard_normal(128))
        v1 = normalize_vector(vec, "red_pitaya")
        v2 = normalize_vector(vec, "keysight_dso")
        # Different hardware → different normalized output
        assert not np.allclose(v1, v2)

    def test_unknown_hardware_falls_back_to_generic(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud-core'))
        from normalizer import normalize_vector
        vec = [0.1] * 128
        # Should not raise, falls back to generic_sdr profile
        result = normalize_vector(vec, "unknown_device_xyz")
        assert len(result) == 128
