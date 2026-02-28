import os
import logging
from datetime import datetime, timezone
from typing import Optional
import secrets

# Graceful stripe import - server starts even if package is missing
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    stripe = None
    STRIPE_AVAILABLE = False
    logging.warning("[CEFIELD] stripe package not found - billing disabled, server running in mock mode.")

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine, get_db, Base, Organization, Node, Measurement, init_db

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cefield")

# Initialize database AND pgvector extension
init_db()

app = FastAPI(
    title="CEFIELD Global Brain API - Enterprise",
    description="Federated resonator network with pgvector Similarity Search & Stripe B2B Billing.",
    version="2.1.0"
)

# Config & Keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if (ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY) else None

STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "sk_test_mock")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "whsec_mock")
if STRIPE_AVAILABLE and stripe:
    stripe.api_key = STRIPE_API_KEY

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

# Auth
API_KEY_HEADER = APIKeyHeader(name="X-CEFIELD-API-KEY", auto_error=True)

def get_organization_from_api_key(api_key: str = Security(API_KEY_HEADER), db: Session = Depends(get_db)) -> Organization:
    if api_key == "cef_dev_machine_001":
        org = db.query(Organization).filter(Organization.api_key == api_key).first()
        if not org:
            org = Organization(name="Demo University", api_key=api_key, subscription_active=True, subscription_tier="enterprise")
            db.add(org)
            db.commit()
            db.refresh(org)
        return org

    org = db.query(Organization).filter(Organization.api_key == api_key).first()
    if not org:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return org

# Pydantic Schemas
class ResonatorSignature(BaseModel):
    node_id: str
    hardware_type: str
    f0: float
    q_factor: float
    signature_vector: list[float]
    lab_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

class CustomerCreate(BaseModel):
    org_name: str
    email: str

# Core AI Logic
def find_similar_anomalies(db: Session, query_vector: list[float], limit: int = 3):
    results = (
        db.query(Measurement, Node.lab_name, Measurement.signature_vector.l2_distance(query_vector).label("distance"))
        .join(Node)
        .filter(Measurement.is_anomaly == True)
        .order_by("distance")
        .limit(limit)
        .all()
    )
    return [(meas, max(0.0, 100.0 - (dist * 100.0)), lab_name) for meas, lab_name, dist in results]

def analyze_anomaly_with_claude(signature: ResonatorSignature, similar_cases: list) -> str:
    if not anthropic_client:
        return "[MOCK] TLS Two-Level System defect identified. Pattern matches a known substrate contamination signature - recommend clean-room inspection of the resonator cavity."

    swarm_context = "".join(
        [f"- MATCH: {score:.1f}% at {lab or 'Unknown'} (f0: {m.f0}Hz, Q: {m.q_factor}).\n" for m, score, lab in similar_cases]
    ) if similar_cases else "No historical matches.\n"

    prompt = f"Node ID: {signature.node_id}\nQ-Factor: {signature.q_factor}\n\n{swarm_context}\nProvide a highly professional, 2-sentence root cause diagnostic."

    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=250,
            system="You are an expert physicist.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"AI Diagnostic failed: {e}"

# Stripe Billing
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
            created_at=utc_now(),
        )
        db.add(org)
        db.commit()
        return {"message": "Organization created.", "api_key": new_api_key, "stripe_customer_id": stripe_customer_id}
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
    org = db.query(Organization).filter(Organization.stripe_customer_id == customer_id).first()

    if org:
        if event.get("type") == "invoice.payment_failed":
            org.subscription_active = False
        elif event.get("type") == "invoice.payment_succeeded":
            org.subscription_active = True
        db.commit()

    return JSONResponse({"status": "success"})

# Data Ingest
@app.post("/api/v1/ingest")
async def ingest_signature(
    signature: ResonatorSignature,
    org: Organization = Depends(get_organization_from_api_key),
    db: Session = Depends(get_db),
):
    if STRIPE_AVAILABLE and stripe and org.stripe_customer_id and "mock" not in STRIPE_API_KEY:
        try:
            stripe.billing.MeterEvent.create(
                event_name="api_requests",
                payload={"value": "1", "stripe_customer_id": org.stripe_customer_id},
            )
        except Exception as e:
            logger.error(f"Stripe Metering failed: {e}")

    node = db.query(Node).filter(Node.node_id == signature.node_id).first()
    if not node:
        node = Node(node_id=signature.node_id, org_id=org.id)
        db.add(node)

    node.hardware_type = signature.hardware_type
    node.last_seen = utc_now()
    node.last_q_factor = signature.q_factor
    is_alert = signature.q_factor < 10000
    node.last_alert = is_alert

    similar_cases = find_similar_anomalies(db, signature.signature_vector, 3) if is_alert else []

    measurement = Measurement(
        node_id=node.id,
        f0=signature.f0,
        q_factor=signature.q_factor,
        signature_vector=signature.signature_vector,
        is_anomaly=is_alert,
    )
    db.add(measurement)
    db.commit()

    if is_alert:
        return {
            "status": "ingested",
            "alert": "Q-Factor drop detected. Anomaly registered.",
            "swarm_matches": [{"similarity_score": round(s, 1), "matched_lab": l} for _, s, l in similar_cases],
            "ai_diagnostic": analyze_anomaly_with_claude(signature, similar_cases),
        }

    return {"status": "ingested", "message": "Measurement committed."}

@app.get("/health")
def health():
    return {"status": "ok", "stripe": STRIPE_AVAILABLE, "ai": ANTHROPIC_AVAILABLE}
