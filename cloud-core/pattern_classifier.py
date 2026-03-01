"""
CEFIELD Pre-Failure Pattern Classifier
========================================
Retroactively labels measurements as 'precursor' patterns when
a confirmed failure event is registered.

This is the mechanism that builds CEFIELD's most valuable asset:
the global pre-failure pattern library. Every confirmed failure
automatically enriches the network's predictive intelligence for
all future labs with similar hardware.

Pattern lifecycle:
  normal → [time passes] → precursor_72h → precursor_24h → failure

The precursor patterns are what enable:
  "This pattern was seen at 5 other labs — they all failed within 24h."
"""
from __future__ import annotations

from datetime import timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from database import Measurement


def classify_precursor_patterns(
    db: Session,
    node_id: int,
    failure_measurement_id: int,
) -> int:
    """
    Triggered when a failure measurement is confirmed.
    Labels the preceding 72h of this node's measurements as precursor patterns.

    Returns: total number of measurements reclassified.
    """
    failure_ts = (
        db.query(Measurement.timestamp)
        .filter(Measurement.id == failure_measurement_id)
        .scalar()
    )
    if not failure_ts:
        return 0

    window_72h = failure_ts - timedelta(hours=72)
    window_24h = failure_ts - timedelta(hours=24)

    # Label 72h–24h window as early precursor
    n1 = (
        db.query(Measurement)
        .filter(
            Measurement.node_id == node_id,
            Measurement.timestamp >= window_72h,
            Measurement.timestamp < window_24h,
            Measurement.pattern_type == "normal",
        )
        .update({"pattern_type": "precursor_72h"}, synchronize_session=False)
    )

    # Label final 24h window as critical precursor (highest predictive value)
    n2 = (
        db.query(Measurement)
        .filter(
            Measurement.node_id == node_id,
            Measurement.timestamp >= window_24h,
            Measurement.timestamp < failure_ts,
            Measurement.pattern_type == "normal",
        )
        .update({"pattern_type": "precursor_24h"}, synchronize_session=False)
    )

    # Label the failure measurement itself
    db.query(Measurement).filter(
        Measurement.id == failure_measurement_id
    ).update({"pattern_type": "failure"}, synchronize_session=False)

    db.commit()
    return n1 + n2


def get_network_precursor_stats(db: Session) -> dict:
    """
    Returns the global distribution of pattern types.
    This is the 'network moat' metric — more precursor patterns = more predictive power.
    """
    stats = (
        db.query(Measurement.pattern_type, func.count(Measurement.id))
        .group_by(Measurement.pattern_type)
        .all()
    )
    return {ptype: count for ptype, count in stats}
