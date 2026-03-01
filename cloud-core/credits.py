"""
CEFIELD Credit System — Metrologie-Intelligence Börse
======================================================
Transforms CEFIELD from a pure SaaS platform into a data marketplace
where labs earn credits by contributing valuable failure patterns
and spend credits consuming network intelligence.

Credit Flow:
  Lab contributes measurement     → +1  credit  (base ingestion)
  Lab confirms failure event      → +50 credits (most valuable data)
  Precursor pattern generated     → +25 credits (retroactive reward)
  Lab queries precursor matches   → -10 credits (intelligence consumption)
  Lab requests AI diagnostic      → -5  credits (compute consumption)

Strategic effect: Labs that document failures honestly get MORE network
intelligence than they consume. This creates a positive-sum incentive
structure where transparency is rewarded.

The credit_balance lives on the Organization model for zero-migration
integration with the existing billing infrastructure.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from database import Organization, CreditTransaction

logger = logging.getLogger("cefield.credits")


# ─── Credit Reward / Cost Table ──────────────────────────────────────────────────
class CreditAction(str, Enum):
    """All actions that affect credit balance."""
    INGEST_NORMAL = "ingest_normal"              # +1  routine measurement
    INGEST_ANOMALY = "ingest_anomaly"            # +3  anomaly (more valuable)
    CONFIRM_FAILURE = "confirm_failure"           # +50 confirmed failure event
    PRECURSOR_GENERATED = "precursor_generated"   # +25 retroactive precursor label
    QUERY_PRECURSOR = "query_precursor"           # -10 consume precursor matches
    AI_DIAGNOSTIC = "ai_diagnostic"              # -5  Claude analysis
    ADMIN_GRANT = "admin_grant"                  # +N  manual admin grant
    SIGNUP_BONUS = "signup_bonus"                # +100 welcome bonus


CREDIT_REWARDS: dict[CreditAction, int] = {
    CreditAction.INGEST_NORMAL:       1,
    CreditAction.INGEST_ANOMALY:      3,
    CreditAction.CONFIRM_FAILURE:     50,
    CreditAction.PRECURSOR_GENERATED: 25,
    CreditAction.QUERY_PRECURSOR:     -10,
    CreditAction.AI_DIAGNOSTIC:       -5,
    CreditAction.ADMIN_GRANT:         0,   # Variable — set at call time
    CreditAction.SIGNUP_BONUS:        100,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Core Credit Operations ──────────────────────────────────────────────────────
def apply_credit(
    db: Session,
    org: Organization,
    action: CreditAction,
    amount_override: Optional[int] = None,
    reference_id: Optional[str] = None,
    note: Optional[str] = None,
) -> CreditTransaction:
    """
    Apply a credit transaction to an organization.

    Args:
        db:              Active SQLAlchemy session
        org:             Organization to credit/debit
        action:          CreditAction enum value
        amount_override: Override default amount (e.g., for ADMIN_GRANT)
        reference_id:    Optional link to measurement/node ID
        note:            Optional human-readable note

    Returns:
        The persisted CreditTransaction record
    """
    amount = amount_override if amount_override is not None else CREDIT_REWARDS[action]

    # Update running balance on Organization
    old_balance = org.credit_balance or 0
    new_balance = old_balance + amount
    org.credit_balance = new_balance

    # Create immutable transaction log entry
    tx = CreditTransaction(
        org_id=org.id,
        action=action.value,
        amount=amount,
        balance_after=new_balance,
        reference_id=reference_id,
        note=note,
        created_at=_utc_now(),
    )
    db.add(tx)
    db.flush()

    logger.info(
        f"[CREDIT] {org.name}: {action.value} → {amount:+d} credits "
        f"(balance: {old_balance} → {new_balance})"
    )
    return tx


def check_sufficient_credits(
    org: Organization,
    action: CreditAction,
) -> tuple[bool, int]:
    """
    Check if organization has enough credits for a debit action.

    IMPORTANT: Earning actions (positive reward) ALWAYS return True —
    they are never blocked by balance. Only debit actions (negative
    reward) require a balance check.

    Returns:
        (has_sufficient, current_balance)
    """
    reward = CREDIT_REWARDS.get(action, 0)
    balance = org.credit_balance or 0

    # Earn / neutral actions are never blocked by balance
    if reward >= 0:
        return (True, balance)

    # Debit action: check if balance covers the cost
    cost = abs(reward)
    return (balance >= cost, balance)


def get_credit_balance(org: Organization) -> dict:
    """Returns current credit state for API response."""
    balance = org.credit_balance or 0
    return {
        "org_id": org.id,
        "org_name": org.name,
        "credit_balance": balance,
        "tier": org.subscription_tier,
        "can_query_precursors": balance >= abs(CREDIT_REWARDS[CreditAction.QUERY_PRECURSOR]),
        "can_request_ai_diagnostic": balance >= abs(CREDIT_REWARDS[CreditAction.AI_DIAGNOSTIC]),
    }


def get_credit_history(
    db: Session,
    org: Organization,
    limit: int = 50,
) -> list[dict]:
    """Returns recent credit transaction history for an organization."""
    txns = (
        db.query(CreditTransaction)
        .filter(CreditTransaction.org_id == org.id)
        .order_by(CreditTransaction.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": tx.id,
            "action": tx.action,
            "amount": tx.amount,
            "balance_after": tx.balance_after,
            "reference_id": tx.reference_id,
            "note": tx.note,
            "timestamp": tx.created_at.isoformat() if tx.created_at else None,
        }
        for tx in txns
    ]


def get_network_credit_stats(db: Session) -> dict:
    """
    Aggregate credit statistics across the entire network.
    Used for /api/v1/network/stats to show marketplace health.
    """
    total_earned = (
        db.query(func.coalesce(func.sum(CreditTransaction.amount), 0))
        .filter(CreditTransaction.amount > 0)
        .scalar()
    )
    total_spent = (
        db.query(func.coalesce(func.sum(func.abs(CreditTransaction.amount)), 0))
        .filter(CreditTransaction.amount < 0)
        .scalar()
    )
    total_orgs_with_balance = (
        db.query(func.count(Organization.id))
        .filter(Organization.credit_balance > 0)
        .scalar()
    )
    total_transactions = db.query(func.count(CreditTransaction.id)).scalar()

    return {
        "total_credits_earned": int(total_earned),
        "total_credits_spent": int(total_spent),
        "net_credits_in_circulation": int(total_earned) - int(total_spent),
        "active_credit_holders": int(total_orgs_with_balance),
        "total_transactions": int(total_transactions),
    }
