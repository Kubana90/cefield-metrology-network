"""
CEFIELD Temporal Baseline Engine
=================================
Replaces naive hardcoded threshold detection with statistical
baseline modelling per node:
  - Rolling mean / std of Q-factor history
  - Linear drift slope (Q-units per day)
  - Z-score anomaly detection
  - Time-to-failure prediction

This is the core algorithmic differentiator vs. all open-source tools.
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from database import Measurement, NodeBaseline

BASELINE_WINDOW: int = 100  # Max recent measurements for baseline
MIN_SAMPLES: int = 10  # Minimum samples before baseline is valid
Z_SCORE_ALERT_THRESHOLD: float = 2.5
CRITICAL_Q_FACTOR_DEFAULT: float = 5_000.0  # Conservative hardware-agnostic fallback


def compute_node_baseline(db: Session, node_id: int, window: int = BASELINE_WINDOW) -> dict:
    """
    Compute statistical fingerprint of a node's Q-factor history.

    Returns a dict with:
      status           : 'ok' | 'insufficient_data'
      n_samples        : int
      needs_warmup     : bool
      mean_q           : float
      std_q            : float
      slope_per_day    : float  (negative = degrading)
      z_score_last     : float  (last measurement vs baseline)
      is_statistical_anomaly : bool
    """
    recent = (
        db.query(Measurement.q_factor, Measurement.timestamp)
        .filter(Measurement.node_id == node_id)
        .order_by(Measurement.timestamp.desc())
        .limit(window)
        .all()
    )

    n = len(recent)
    if n < MIN_SAMPLES:
        return {"status": "insufficient_data", "n_samples": n, "needs_warmup": True}

    values = np.array([r.q_factor for r in recent], dtype=np.float64)
    timestamps = np.array(
        [r.timestamp.timestamp() if r.timestamp else datetime.now(UTC).timestamp() for r in recent],
        dtype=np.float64,
    )

    mean_q = float(np.mean(values))
    std_q = float(np.std(values))
    effective_std = std_q if std_q > 1e-9 else 1.0

    # Linear drift slope via least-squares fit
    t_days = (timestamps - timestamps.min()) / 86_400.0
    if t_days.max() > 0:
        coeffs = np.polyfit(t_days, values, 1)
        slope_per_day = float(coeffs[0])
    else:
        slope_per_day = 0.0

    z_score_last = float((values[0] - mean_q) / effective_std)
    is_anomaly = abs(z_score_last) > Z_SCORE_ALERT_THRESHOLD

    return {
        "status": "ok",
        "n_samples": n,
        "needs_warmup": False,
        "mean_q": mean_q,
        "std_q": std_q,
        "slope_per_day": slope_per_day,
        "z_score_last": z_score_last,
        "is_statistical_anomaly": is_anomaly,
    }


def predict_time_to_failure(
    baseline: dict,
    critical_threshold: float = CRITICAL_Q_FACTOR_DEFAULT,
) -> dict | None:
    """
    Given a computed baseline with slope_per_day, estimate when
    the Q-factor will cross the critical threshold.

    Returns None if the device is stable or data is insufficient.
    """
    if baseline.get("needs_warmup") or baseline.get("status") != "ok":
        return None

    slope = baseline["slope_per_day"]
    if slope >= 0.0:
        return None  # Stable or improving — no failure predicted

    mean_q = baseline["mean_q"]
    if mean_q <= critical_threshold:
        return {"days_to_failure": 0, "confidence": "high", "status": "already_critical"}

    days_remaining = (mean_q - critical_threshold) / abs(slope)
    n = baseline["n_samples"]
    confidence = "high" if n >= 50 else "medium" if n >= 20 else "low"

    return {
        "days_to_failure": round(days_remaining, 1),
        "confidence": confidence,
        "slope_per_day": round(slope, 2),
        "current_mean_q": round(mean_q, 1),
        "critical_threshold": critical_threshold,
        "status": "degrading",
    }


def update_cached_baseline(db: Session, node_id: int, baseline: dict) -> None:
    """Persist computed baseline to NodeBaseline cache table."""
    if baseline.get("status") != "ok":
        return
    cached = db.query(NodeBaseline).filter(NodeBaseline.node_id == node_id).first()
    if not cached:
        cached = NodeBaseline(node_id=node_id)
        db.add(cached)
    cached.mean_q = baseline["mean_q"]
    cached.std_q = baseline["std_q"]
    cached.slope_per_day = baseline["slope_per_day"]
    cached.n_samples = baseline["n_samples"]
    cached.updated_at = datetime.now(UTC)
    db.commit()
