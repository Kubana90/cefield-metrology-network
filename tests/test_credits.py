"""
CEFIELD Credit System — Test Suite
====================================
Comprehensive tests for the Metrologie-Intelligence credit engine.
"""
import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from credits import (
    CreditAction,
    CREDIT_REWARDS,
    apply_credit,
    check_sufficient_credits,
    get_credit_balance,
    get_credit_history,
)


def _make_org(name="Test Lab", balance=100, tier="freemium"):
    """Create a mock Organization object."""
    org = MagicMock()
    org.id = 1
    org.name = name
    org.credit_balance = balance
    org.subscription_tier = tier
    return org


def _make_db():
    """Create a mock DB session."""
    db = MagicMock()
    db.add = MagicMock()
    db.flush = MagicMock()
    return db


class TestCreditRewardTable:
    """Validate the credit reward/cost table is correctly configured."""

    def test_ingest_normal_earns_credits(self):
        assert CREDIT_REWARDS[CreditAction.INGEST_NORMAL] == 1

    def test_ingest_anomaly_earns_more(self):
        assert CREDIT_REWARDS[CreditAction.INGEST_ANOMALY] == 3
        assert CREDIT_REWARDS[CreditAction.INGEST_ANOMALY] > CREDIT_REWARDS[CreditAction.INGEST_NORMAL]

    def test_confirm_failure_highest_reward(self):
        assert CREDIT_REWARDS[CreditAction.CONFIRM_FAILURE] == 50

    def test_precursor_generated_reward(self):
        assert CREDIT_REWARDS[CreditAction.PRECURSOR_GENERATED] == 25

    def test_query_precursor_costs_credits(self):
        assert CREDIT_REWARDS[CreditAction.QUERY_PRECURSOR] < 0
        assert CREDIT_REWARDS[CreditAction.QUERY_PRECURSOR] == -10

    def test_ai_diagnostic_costs_credits(self):
        assert CREDIT_REWARDS[CreditAction.AI_DIAGNOSTIC] < 0
        assert CREDIT_REWARDS[CreditAction.AI_DIAGNOSTIC] == -5

    def test_signup_bonus(self):
        assert CREDIT_REWARDS[CreditAction.SIGNUP_BONUS] == 100


class TestApplyCredit:
    """Test core credit application logic."""

    def test_apply_ingest_credit(self):
        org = _make_org(balance=100)
        db = _make_db()
        tx = apply_credit(db, org, CreditAction.INGEST_NORMAL)
        assert org.credit_balance == 101
        assert tx.amount == 1
        assert tx.balance_after == 101

    def test_apply_anomaly_credit(self):
        org = _make_org(balance=100)
        db = _make_db()
        tx = apply_credit(db, org, CreditAction.INGEST_ANOMALY)
        assert org.credit_balance == 103

    def test_apply_debit_reduces_balance(self):
        org = _make_org(balance=100)
        db = _make_db()
        tx = apply_credit(db, org, CreditAction.QUERY_PRECURSOR)
        assert org.credit_balance == 90
        assert tx.amount == -10

    def test_apply_failure_confirmation(self):
        org = _make_org(balance=50)
        db = _make_db()
        tx = apply_credit(db, org, CreditAction.CONFIRM_FAILURE)
        assert org.credit_balance == 100

    def test_apply_admin_grant_with_override(self):
        org = _make_org(balance=0)
        db = _make_db()
        tx = apply_credit(
            db, org, CreditAction.ADMIN_GRANT,
            amount_override=500,
            note="Partnership bonus",
        )
        assert org.credit_balance == 500
        assert tx.amount == 500
        assert tx.note == "Partnership bonus"

    def test_balance_can_go_negative(self):
        """Edge case: balance CAN go negative (enforcement is at check level)."""
        org = _make_org(balance=5)
        db = _make_db()
        tx = apply_credit(db, org, CreditAction.QUERY_PRECURSOR)
        assert org.credit_balance == -5

    def test_reference_id_stored(self):
        org = _make_org(balance=100)
        db = _make_db()
        tx = apply_credit(
            db, org, CreditAction.INGEST_NORMAL,
            reference_id="measurement:42",
        )
        assert tx.reference_id == "measurement:42"

    def test_multiple_transactions_accumulate(self):
        org = _make_org(balance=100)
        db = _make_db()
        apply_credit(db, org, CreditAction.INGEST_NORMAL)     # +1 → 101
        apply_credit(db, org, CreditAction.INGEST_NORMAL)     # +1 → 102
        apply_credit(db, org, CreditAction.INGEST_ANOMALY)    # +3 → 105
        apply_credit(db, org, CreditAction.AI_DIAGNOSTIC)     # -5 → 100
        assert org.credit_balance == 100


class TestCheckSufficientCredits:
    """Test pre-flight credit balance checks."""

    def test_sufficient_for_precursor_query(self):
        org = _make_org(balance=50)
        sufficient, balance = check_sufficient_credits(org, CreditAction.QUERY_PRECURSOR)
        assert sufficient is True
        assert balance == 50

    def test_insufficient_for_precursor_query(self):
        org = _make_org(balance=5)
        sufficient, balance = check_sufficient_credits(org, CreditAction.QUERY_PRECURSOR)
        assert sufficient is False

    def test_exactly_sufficient(self):
        org = _make_org(balance=10)
        sufficient, _ = check_sufficient_credits(org, CreditAction.QUERY_PRECURSOR)
        assert sufficient is True

    def test_zero_balance(self):
        org = _make_org(balance=0)
        sufficient, _ = check_sufficient_credits(org, CreditAction.AI_DIAGNOSTIC)
        assert sufficient is False

    def test_earn_action_always_sufficient(self):
        """Credit-earning actions should always show as 'sufficient'."""
        org = _make_org(balance=0)
        sufficient, _ = check_sufficient_credits(org, CreditAction.INGEST_NORMAL)
        assert sufficient is True


class TestGetCreditBalance:
    """Test credit balance API response formatting."""

    def test_balance_response_structure(self):
        org = _make_org(balance=100)
        result = get_credit_balance(org)
        assert "credit_balance" in result
        assert "can_query_precursors" in result
        assert "can_request_ai_diagnostic" in result
        assert result["credit_balance"] == 100

    def test_rich_lab_can_do_everything(self):
        org = _make_org(balance=1000)
        result = get_credit_balance(org)
        assert result["can_query_precursors"] is True
        assert result["can_request_ai_diagnostic"] is True

    def test_poor_lab_limited(self):
        org = _make_org(balance=3)
        result = get_credit_balance(org)
        assert result["can_query_precursors"] is False
        assert result["can_request_ai_diagnostic"] is False

    def test_none_balance_treated_as_zero(self):
        org = _make_org(balance=None)
        org.credit_balance = None
        result = get_credit_balance(org)
        assert result["credit_balance"] == 0
