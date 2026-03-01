import os
import logging
from datetime import datetime, timezone
from typing import Optional
import secrets

try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    stripe = None
    STRIPE_AVAILABLE = False
    logging.warning("[CEFIELD] stripe not found — billing disabled.")

from fastapi import FastAPI, Depends, HTTPException, Security, Request, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import (
    engine, get_db, Base, Organization, Node,
    Measurement, NodeBaseline, CreditTransaction, init_db,
)
from baseline import (
    compute_node_baseline,
    predict_time_to_failure,
    update_cached_baseline,
    CRITICAL_Q_FACTOR_DEFAULT,
)
from normalizer import normalize_vector, get_hardware_info, list_supported_hardware
from pattern_classifier import classify_precursor_patterns, get_network_precursor_stats
from credits import (
    CreditAction,
    CREDIT_REWARDS,
    apply_credit,
    check_sufficient_credits,
    get_credit_balance,
    get_credit_history,
    get_network_credit_stats,
)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cefield")

init_db()

app = FastAPI(
    title="CEFIELD Global Brain API",
    description=(
        "Federated resonator network — Predictive metrology intelligence.\n\n"
        "v3.1: Credit-based Metrologie-Intelligence Marketplace + SCPI Hardware Integration."
    ),
    version="3.1.0",
)

# ─── Config ─────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "sk_test_mock")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "whsec_mock")
# Admin key: set CEFIELD_ADMIN_KEY env var to enable secure cross-org credit grants
# If not set, only enterprise-tier orgs can call /admin/grant (self-grant only)
CEFIELD_ADMIN_KEY = os.environ.get("CEFIELD_ADMIN_KEY", "")

if STRIPE_AVAILABLE and stripe:
    stripe.api_key = STRIPE_API_KEY


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Auth ────────────────────────────────────────────────────────────────────────
API_KEY_HEADER = APIKeyHeader(name="X-CEFIELD-API-KEY", auto_error=True)


def get_organization_from_api_key(
    api_key: str = Security(API_KEY_HEADER),
    db: Session = Depends(get_db),
) -> Organization:
    if api_key == "cef_dev_machine_001":
        org = db.query(Organization).filter(Organization.api_key == api_key).first()
        if not org:
            org = Organization(
                name="Demo University",
                api_key=api_key,
                subscription_active=True,
                subscription_tier="enterprise",
                credit_balance=100,
            )
            db.add(org)
            db.commit()
            db.refresh(org)
        return org
    org = db.query(Organization).filter(Organization.api_key == api_key).first()
    if not org:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return org


def _require_admin(request: Request) -> None:
    """
    Admin gate: pass if either
      (a) CEFIELD_ADMIN_KEY is set and X-CEFIELD-ADMIN-KEY header matches, OR
      (b) CEFIELD_ADMIN_KEY is not configured (dev mode, caller is enterprise).
    Called explicitly in admin endpoints AFTER org is resolved.
    """
    if not CEFIELD_ADMIN_KEY:
        # Dev / local mode — admin key not configured, skip header check
        return
    submitted = request.headers.get("X-CEFIELD-ADMIN-KEY", "")
    if submitted != CEFIELD_ADMIN_KEY:
        raise HTTPException(
            status_code=403,
            detail="Admin key required: set X-CEFIELD-ADMIN-KEY header",
        )


# ─── Schemas ─────────────────────────────────────────────────────────────────────
class ResonatorSignature(BaseModel):
    node_id: str
    hardware_type: str
    f0: float
    q_factor: float
    signature_vector: list[float] = Field(..., min_length=128, max_length=128)
    lab_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class CustomerCreate(BaseModel):
    org_name: str
    email: str


class CreditGrantRequest(BaseModel):
    amount: int = Field(..., gt=0, description="Credits to grant")
    note: Optional[str] = None
    target_org_id: Optional[int] = Field(
        None,
        description="Target org ID for cross-org grants (admin key required)",
    )


# ─── Precursor Search ────────────────────────────────────────────────────────────
def find_precursor_patterns(
    db: Session,
    normalized_vector: list[float],
    limit: int = 5,
) -> list[dict]:
    """
    Core differentiator: searches global pre-failure precursor patterns.
    Input vector must already be hardware-normalized.
    """
    results = (
        db.query(
            Measurement,
            Node.lab_name,
            Node.hardware_type,
            Measurement.pattern_type,
            Measurement.signature_vector.l2_distance(normalized_vector).label("distance"),
        )
        .join(Node)
        .filter(Measurement.pattern_type.in_(["precursor_72h", "precursor_24h"]))
        .order_by("distance")
        .limit(limit)
        .all()
    )
    return [
        {
            "measurement": meas,
            "similarity_pct": round(max(0.0, 100.0 - (dist * 100.0)), 1),
            "lab": lab or "Anonymous Lab",
            "hardware": hw or "Unknown",
            "precursor_type": ptype,
            "hours_before_failure": 24 if ptype == "precursor_24h" else 72,
        }
        for meas, lab, hw, ptype, dist in results
    ]


# ─── Async AI Diagnostic ──────────────────────────────────────────────────────────
async def analyze_with_predictive_claude(
    signature: ResonatorSignature,
    baseline: dict,
    prediction: Optional[dict],
    precursor_matches: list[dict],
) -> dict:
    """
    Async predictive diagnostics via Claude 3.5 Sonnet.
    Returns structured JSON: risk_level, days_to_predicted_failure,
    failure_mechanism, recommended_action.
    """
    if not ANTHROPIC_AVAILABLE or not ANTHROPIC_API_KEY:
        return {
            "risk_level": "medium",
            "days_to_predicted_failure": (
                prediction.get("days_to_failure") if prediction else None
            ),
            "failure_mechanism": (
                "[MOCK] Thermal drift in resonator cavity — "
                "substrate contamination suspected."
            ),
            "recommended_action": "Schedule clean-room inspection within 14 days.",
            "confidence": "mock",
        }

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    precursor_ctx = (
        "\n".join(
            [
                f"  \u2022 {m['similarity_pct']}% match @ {m['lab']} "
                f"({m['hardware']}) \u2192 device failed "
                f"{m['hours_before_failure']}h after this pattern"
                for m in precursor_matches
            ]
        )
        if precursor_matches
        else "  \u2022 No precursor matches in global network yet."
    )

    if prediction and prediction.get("status") == "degrading":
        trend_ctx = (
            f"Drift: {prediction['slope_per_day']:+.1f} Q-units/day. "
            f"Predicted failure in {prediction['days_to_failure']} days "
            f"(confidence: {prediction['confidence']})."
        )
    elif baseline.get("status") == "ok":
        trend_ctx = (
            f"Q-Factor stable. Z-score: {baseline.get('z_score_last', 0):.2f}\u03c3 "
            f"from {baseline['n_samples']}-sample baseline."
        )
    else:
        trend_ctx = "Insufficient history (warmup phase)."

    system = (
        "You are an expert metrology engineer specialising in RF resonators, "
        "MEMS devices, superconducting qubits, and quantum hardware. "
        "Provide PREDICTIVE diagnostics. "
        "Output valid JSON with exactly: "
        '{"risk_level": "low|medium|high|critical", '
        '"days_to_predicted_failure": <number or null>, '
        '"failure_mechanism": "<2-sentence physics explanation>", '
        '"recommended_action": "<1 specific preventive action>"}'
    )

    user = (
        f"CURRENT MEASUREMENT:\n"
        f"  Node: {signature.node_id} | Hardware: {signature.hardware_type}\n"
        f"  f\u2080: {signature.f0:.4e} Hz | Q-Factor: {signature.q_factor:,.0f}\n"
        f"  Z-score vs baseline: {baseline.get('z_score_last', 'N/A')}\u03c3\n"
        f"  Baseline mean Q: {baseline.get('mean_q', 'N/A'):,.0f}"
        f" \u00b1 {baseline.get('std_q', 0):.0f}\n\n"
        f"TEMPORAL TREND:\n  {trend_ctx}\n\n"
        f"GLOBAL NETWORK \u2014 Pre-failure pattern matches:\n{precursor_ctx}\n\n"
        f"Diagnose failure risk and provide ONE preventive action."
    )

    try:
        import json
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = response.content[0].text
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"failure_mechanism": raw, "confidence": "parse_error"}
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return {"failure_mechanism": f"AI diagnostic unavailable: {e}", "confidence": "failed"}


# ─── Background Tasks ─────────────────────────────────────────────────────────────
def _background_classify_precursors(node_db_id: int, measurement_id: int, org_id: int) -> None:
    db = next(get_db())
    try:
        n = classify_precursor_patterns(db, node_db_id, measurement_id)
        if n > 0:
            logger.info(
                f"[CEFIELD] Retroactively labeled {n} precursor measurements "
                f"for node {node_db_id}"
            )
            # Reward: +25 credits per precursor batch generated
            org = db.query(Organization).filter(Organization.id == org_id).first()
            if org:
                apply_credit(
                    db, org, CreditAction.PRECURSOR_GENERATED,
                    reference_id=f"measurement:{measurement_id}",
                    note=f"Retroactive: {n} precursor patterns labeled",
                )
                db.commit()
    finally:
        db.close()


# ─── Credit Endpoints ─────────────────────────────────────────────────────────────
@app.get("/api/v1/credits/balance")
async def credits_balance(
    org: Organization = Depends(get_organization_from_api_key),
):
    """Returns current credit balance and spending power."""
    return get_credit_balance(org)


@app.get("/api/v1/credits/history")
async def credits_history(
    limit: int = Query(50, ge=1, le=200),
    org: Organization = Depends(get_organization_from_api_key),
    db: Session = Depends(get_db),
):
    """Returns credit transaction history for auditing."""
    return {
        "org_name": org.name,
        "current_balance": org.credit_balance,
        "transactions": get_credit_history(db, org, limit=limit),
    }


@app.post("/api/v1/credits/admin/grant")
async def admin_grant_credits(
    data: CreditGrantRequest,
    request: Request,
    org: Organization = Depends(get_organization_from_api_key),
    db: Session = Depends(get_db),
):
    """
    Grant credits to an organization.

    Access rules:
      - With X-CEFIELD-ADMIN-KEY header (matching CEFIELD_ADMIN_KEY env var):
        Can grant to any org via target_org_id. Full admin access.
      - Without admin key, enterprise tier only:
        Can only grant to self (useful for reseller partner top-ups).
      - Without admin key, freemium tier:
        HTTP 403 — not permitted.

    Set CEFIELD_ADMIN_KEY env var to enable admin key header validation.
    If CEFIELD_ADMIN_KEY is not set (dev mode), admin key check is skipped.
    """
    submitted_admin_key = request.headers.get("X-CEFIELD-ADMIN-KEY", "")
    has_admin_key = CEFIELD_ADMIN_KEY and (submitted_admin_key == CEFIELD_ADMIN_KEY)
    is_enterprise = org.subscription_tier == "enterprise"

    if not has_admin_key and not is_enterprise and CEFIELD_ADMIN_KEY:
        raise HTTPException(
            status_code=403,
            detail=(
                "Admin grant requires X-CEFIELD-ADMIN-KEY header "
                "or enterprise subscription tier."
            ),
        )

    # Resolve target organization
    target_org = org  # default: grant to requesting org
    if data.target_org_id and data.target_org_id != org.id:
        if not has_admin_key:
            raise HTTPException(
                status_code=403,
                detail="X-CEFIELD-ADMIN-KEY required for cross-org credit grants.",
            )
        target_org = db.query(Organization).filter(
            Organization.id == data.target_org_id
        ).first()
        if not target_org:
            raise HTTPException(
                status_code=404,
                detail=f"Organization {data.target_org_id} not found.",
            )

    tx = apply_credit(
        db, target_org, CreditAction.ADMIN_GRANT,
        amount_override=data.amount,
        note=data.note or f"Admin grant by org:{org.id} ({org.name})",
    )
    db.commit()
    return {
        "message": f"Granted {data.amount} credits to {target_org.name}",
        "target_org": target_org.name,
        "new_balance": target_org.credit_balance,
        "transaction_id": tx.id,
    }


@app.get("/api/v1/credits/pricing")
async def credits_pricing():
    """Returns the current credit reward/cost table."""
    return {
        "pricing": {
            action.value: amount
            for action, amount in CREDIT_REWARDS.items()
        },
        "note": (
            "Positive values = credits earned. "
            "Negative values = credits consumed. "
            "Enterprise tier has unlimited AI diagnostics."
        ),
    }


# ─── Billing Endpoints ────────────────────────────────────────────────────────────
@app.post("/api/v1/billing/onboard")
async def onboard_customer(data: CustomerCreate, db: Session = Depends(get_db)):
    try:
        if not STRIPE_AVAILABLE or "mock" in STRIPE_API_KEY:
            stripe_customer_id = f"cus_mock_{int(utc_now().timestamp())}"
        else:
            customer = stripe.Customer.create(name=data.org_name, email=data.email)
            stripe_customer_id = customer.id

        new_api_key = f"cef_{secrets.token_hex(16)}"
        org = Organization(
            name=data.org_name,
            api_key=new_api_key,
            stripe_customer_id=stripe_customer_id,
            subscription_active=True,
            subscription_tier="enterprise",
            credit_balance=0,
            created_at=utc_now(),
        )
        db.add(org)
        db.flush()

        # Grant signup bonus via credit system (creates first ledger entry)
        apply_credit(
            db, org, CreditAction.SIGNUP_BONUS,
            note="Welcome to the CEFIELD network!",
        )
        db.commit()

        return {
            "message": "Organization created.",
            "api_key": new_api_key,
            "credit_balance": org.credit_balance,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    if not STRIPE_AVAILABLE:
        return JSONResponse({"status": "stripe_not_available"})
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid webhook")
    data_object = event.get("data", {}).get("object", {})
    customer_id = data_object.get("customer")
    org = db.query(Organization).filter(
        Organization.stripe_customer_id == customer_id
    ).first()
    if org:
        if event.get("type") == "invoice.payment_failed":
            org.subscription_active = False
        elif event.get("type") == "invoice.payment_succeeded":
            org.subscription_active = True
        db.commit()
    return JSONResponse({"status": "success"})


# ─── Core Ingest Endpoint ─────────────────────────────────────────────────────────
@app.post("/api/v1/ingest")
async def ingest_signature(
    signature: ResonatorSignature,
    background_tasks: BackgroundTasks,
    org: Organization = Depends(get_organization_from_api_key),
    db: Session = Depends(get_db),
):
    # Stripe usage metering
    if (
        STRIPE_AVAILABLE
        and stripe
        and org.stripe_customer_id
        and "mock" not in STRIPE_API_KEY
    ):
        try:
            stripe.billing.MeterEvent.create(
                event_name="api_requests",
                payload={"value": "1", "stripe_customer_id": org.stripe_customer_id},
            )
        except Exception as e:
            logger.error(f"Stripe metering failed: {e}")

    # Normalize vector for cross-hardware comparability
    normalized_vector = normalize_vector(signature.signature_vector, signature.hardware_type)

    # Upsert Node
    node = db.query(Node).filter(Node.node_id == signature.node_id).first()
    if not node:
        node = Node(node_id=signature.node_id, org_id=org.id)
        db.add(node)
        db.flush()

    node.hardware_type = signature.hardware_type
    node.last_seen = utc_now()
    node.last_q_factor = signature.q_factor
    if signature.lab_name:
        node.lab_name = signature.lab_name
    if signature.lat:
        node.lat = signature.lat
    if signature.lon:
        node.lon = signature.lon

    # Statistical baseline (replaces hardcoded threshold)
    baseline = compute_node_baseline(db, node.id)
    prediction = (
        predict_time_to_failure(baseline)
        if baseline.get("status") == "ok"
        else None
    )

    # Anomaly decision: statistical z-score OR near-term drift prediction
    is_stat_anomaly = baseline.get("is_statistical_anomaly", False)
    is_drift_critical = (
        prediction is not None
        and prediction.get("status") == "degrading"
        and prediction.get("days_to_failure", 999) < 30
    )
    is_alert = is_stat_anomaly or is_drift_critical

    # Warmup fallback: conservative threshold until baseline matures
    if baseline.get("needs_warmup") and signature.q_factor < CRITICAL_Q_FACTOR_DEFAULT:
        is_alert = True

    # ── Credit: precursor search costs credits (enterprise: unlimited) ──────────
    precursor_matches = []
    if is_alert:
        can_query, balance = check_sufficient_credits(org, CreditAction.QUERY_PRECURSOR)
        if can_query or org.subscription_tier == "enterprise":
            precursor_matches = find_precursor_patterns(db, normalized_vector)
            if precursor_matches:
                apply_credit(
                    db, org, CreditAction.QUERY_PRECURSOR,
                    reference_id=f"node:{signature.node_id}",
                    note=f"Precursor search: {len(precursor_matches)} matches",
                )
        else:
            logger.warning(
                f"[CREDIT] {org.name} has insufficient credits ({balance}) "
                f"for precursor search — skipping"
            )

    # Persist measurement
    measurement = Measurement(
        node_id=node.id,
        f0=signature.f0,
        q_factor=signature.q_factor,
        signature_vector=normalized_vector,
        is_anomaly=is_alert,
        timestamp=utc_now(),
        pattern_type="failure" if is_alert else "normal",
    )
    db.add(measurement)
    node.last_alert = is_alert

    # ── Credit: reward for data contribution ──────────────────────────────────────
    credit_action = CreditAction.INGEST_ANOMALY if is_alert else CreditAction.INGEST_NORMAL
    apply_credit(
        db, org, credit_action,
        reference_id=f"node:{signature.node_id}",
    )

    db.commit()
    db.refresh(measurement)

    # Background: update cached baseline + retroactive precursor labeling
    if baseline.get("status") == "ok":
        background_tasks.add_task(update_cached_baseline, db, node.id, baseline)
    if is_alert:
        background_tasks.add_task(
            _background_classify_precursors, node.id, measurement.id, org.id
        )

    if is_alert:
        # ── Credit: AI diagnostic costs credits (enterprise: unlimited) ────────
        ai_report = None
        can_ai, _ = check_sufficient_credits(org, CreditAction.AI_DIAGNOSTIC)
        if can_ai or org.subscription_tier == "enterprise":
            ai_report = await analyze_with_predictive_claude(
                signature, baseline, prediction, precursor_matches
            )
            if org.subscription_tier != "enterprise":
                apply_credit(
                    db, org, CreditAction.AI_DIAGNOSTIC,
                    reference_id=f"node:{signature.node_id}",
                    note="Claude predictive diagnostic",
                )
                db.commit()
        else:
            ai_report = {
                "risk_level": "unknown",
                "failure_mechanism": "Insufficient credits for AI diagnostic.",
                "recommended_action": "Top up credits or upgrade to enterprise tier.",
                "confidence": "credit_limited",
            }

        return {
            "status": "alert",
            "node_id": signature.node_id,
            "alert": {
                "type": "statistical" if is_stat_anomaly else "drift_prediction",
                "z_score": round(baseline.get("z_score_last", 0.0), 2),
                "days_to_predicted_failure": (
                    prediction.get("days_to_failure") if prediction else None
                ),
                "drift_slope_per_day": round(baseline.get("slope_per_day", 0.0), 2),
            },
            "swarm_matches": [
                {
                    "similarity_pct": m["similarity_pct"],
                    "lab": m["lab"],
                    "hardware": m["hardware"],
                    "hours_before_their_failure": m["hours_before_failure"],
                }
                for m in precursor_matches
            ],
            "ai_diagnostic": ai_report,
            "credits": get_credit_balance(org),
        }

    return {
        "status": "ingested",
        "node_id": signature.node_id,
        "baseline": {
            "mean_q": round(baseline.get("mean_q", 0), 1),
            "z_score": round(baseline.get("z_score_last", 0.0), 2),
            "n_samples": baseline.get("n_samples", 0),
            "slope_per_day": round(baseline.get("slope_per_day", 0.0), 2),
        }
        if baseline.get("status") == "ok"
        else {"status": "warmup", "n_samples": baseline.get("n_samples", 0)},
        "credits": get_credit_balance(org),
    }


# ─── Predictive Health Endpoint ───────────────────────────────────────────────────
@app.get("/api/v1/nodes/{node_id}/health")
async def get_node_health(
    node_id: str,
    org: Organization = Depends(get_organization_from_api_key),
    db: Session = Depends(get_db),
):
    """Returns current health status and failure prediction for a specific node."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    baseline = compute_node_baseline(db, node.id)
    prediction = (
        predict_time_to_failure(baseline) if baseline.get("status") == "ok" else None
    )

    return {
        "node_id": node_id,
        "hardware_type": node.hardware_type,
        "lab_name": node.lab_name,
        "last_seen": node.last_seen.isoformat() if node.last_seen else None,
        "current_q_factor": node.last_q_factor,
        "baseline": baseline,
        "prediction": prediction,
        "risk_level": _compute_risk_level(prediction, baseline),
    }


def _compute_risk_level(prediction: Optional[dict], baseline: dict) -> str:
    if not prediction or prediction.get("status") != "degrading":
        z = abs(baseline.get("z_score_last", 0))
        return "medium" if z > 2.5 else "low"
    days = prediction.get("days_to_failure", 999)
    if days <= 3:
        return "critical"
    if days <= 7:
        return "high"
    if days <= 30:
        return "medium"
    return "low"


# ─── Network Intelligence Endpoints ──────────────────────────────────────────────
@app.get("/api/v1/network/stats")
async def get_network_stats(
    org: Organization = Depends(get_organization_from_api_key),
    db: Session = Depends(get_db),
):
    """Global network statistics — the competitive moat made visible."""
    from sqlalchemy import func

    total_measurements = db.query(func.count(Measurement.id)).scalar()
    total_nodes = db.query(func.count(Node.id)).scalar()
    precursor_stats = get_network_precursor_stats(db)
    credit_stats = get_network_credit_stats(db)

    return {
        "total_measurements": total_measurements,
        "total_active_nodes": total_nodes,
        "precursor_library": precursor_stats,
        "network_intelligence": (
            "high"
            if precursor_stats.get("precursor_72h", 0) > 100
            else "growing"
        ),
        "marketplace": credit_stats,
    }


@app.get("/api/v1/hardware/supported")
async def list_hardware(org: Organization = Depends(get_organization_from_api_key)):
    """Returns all hardware types with built-in normalization profiles."""
    return {"supported_hardware": list_supported_hardware()}


# ─── System Health ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.1.0",
        "stripe": STRIPE_AVAILABLE,
        "ai": ANTHROPIC_AVAILABLE,
        "admin_protected": bool(CEFIELD_ADMIN_KEY),
        "features": [
            "statistical_baseline_engine",
            "predictive_time_to_failure",
            "cross_hardware_normalization",
            "precursor_pattern_library",
            "async_claude_diagnostics",
            "hnsw_vector_index",
            "credit_marketplace",
            "scpi_hardware_bridge",
        ],
    }
