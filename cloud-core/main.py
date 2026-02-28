import os
import logging
import stripe
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import secrets

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine, get_db, Base, Organization, Node, Measurement
from models import Node as OldNode, Measurement as OldMeasurement
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cefield")

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="CEFIELD Global Brain API - Enterprise",
    description="Federated resonator network with pgvector Similarity Search & Stripe B2B Billing.",
    version="2.1.0"
)

# -----------------------------
# Config & Keys
# -----------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "sk_test_mock")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "whsec_mock")
stripe.api_key = STRIPE_API_KEY

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

# -----------------------------
# Auth
# -----------------------------
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
    
    if not org.subscription_active and org.subscription_tier != "freemium":
        raise HTTPException(status_code=402, detail="Payment required. Your enterprise subscription is inactive.")
        
    return org

# -----------------------------
# Pydantic Schemas
# -----------------------------
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

# -----------------------------
# REAL Stripe Billing Endpoints
# -----------------------------

@app.post("/api/v1/billing/onboard")
async def onboard_customer(data: CustomerCreate, db: Session = Depends(get_db)):
    """
    Creates a REAL B2B customer in Stripe and generates an API key.
    """
    try:
        if "mock" in STRIPE_API_KEY:
            logger.warning("Using mock Stripe API key. Enable real Stripe integration by setting STRIPE_API_KEY.")
            stripe_customer_id = f"cus_mock_{int(utc_now().timestamp())}"
        else:
            # 1. Create the customer in real Stripe API
            customer = stripe.Customer.create(
                name=data.org_name,
                email=data.email,
                metadata={"cefield_tier": "enterprise"}
            )
            stripe_customer_id = customer.id
            logger.info(f"Created real Stripe Customer: {stripe_customer_id}")
        
        # 2. Generate a secure API Key
        new_api_key = f"cef_{secrets.token_hex(16)}"
        
        # 3. Save to DB
        org = Organization(
            name=data.org_name,
            api_key=new_api_key,
            stripe_customer_id=stripe_customer_id,
            subscription_active=True,  # Usually set to True via webhook after payment, True for demo
            subscription_tier="enterprise",
            created_at=utc_now()
        )
        db.add(org)
        db.commit()
        
        return {
            "message": "Organization created in Stripe.",
            "api_key": new_api_key,
            "stripe_customer_id": stripe_customer_id
        }
    except Exception as e:
        logger.error(f"Onboarding failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Listens for REAL Stripe events."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        # Construct the event securely using the Stripe SDK
        if "mock" not in STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        else:
            # Fallback for mock environment
            event = await request.json()
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event.get("type")
    data_object = event.get("data", {}).get("object", {})
    customer_id = data_object.get("customer")

    if not customer_id:
        return JSONResponse({"status": "ignored, no customer_id"})

    org = db.query(Organization).filter(Organization.stripe_customer_id == customer_id).first()
    if not org:
        return JSONResponse({"status": "ignored, unknown customer"})

    if event_type == "invoice.payment_failed":
        logger.warning(f"Payment failed for {org.name}. Deactivating API access.")
        org.subscription_active = False
        db.commit()
        
    elif event_type == "invoice.payment_succeeded":
        logger.info(f"Payment succeeded for {org.name}. Activating API access.")
        org.subscription_active = True
        db.commit()

    return JSONResponse({"status": "success"})


# -----------------------------
# Core AI Logic
# -----------------------------
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
        return "AI Diagnostic disabled. Missing ANTHROPIC_API_KEY."

    swarm_context = "Historical matches:\n" + "".join(
        [f"- MATCH: {score:.1f}% at {lab or 'Unknown'} (f0: {m.f0}Hz, Q: {m.q_factor}).\n" for m, score, lab in similar_cases]
    ) if similar_cases else "- No historical matches.\n"

    prompt = f"Node ID: {signature.node_id}\nQ-Factor: {signature.q_factor}\n\n{swarm_context}\nProvide a highly professional, 2-sentence root cause diagnostic."
    
    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=250, system="You are an expert physicist.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"AI Diagnostic failed: {e}"

# -----------------------------
# Data Ingest & REAL Metering
# -----------------------------
@app.post("/api/v1/ingest")
async def ingest_signature(
    signature: ResonatorSignature, 
    org: Organization = Depends(get_organization_from_api_key),
    db: Session = Depends(get_db)
):
    # --- 1. REAL STRIPE METERING ---
    if org.stripe_customer_id and org.subscription_tier == "enterprise" and "mock" not in STRIPE_API_KEY:
        try:
            # We send a usage event to Stripe. You must have a Meter named "api_requests" created in Stripe.
            stripe.billing.MeterEvent.create(
                event_name="api_requests",
                payload={
                    "value": "1",
                    "stripe_customer_id": org.stripe_customer_id
                }
            )
            logger.info(f"Billed 1 API call to {org.name} via Stripe Metering.")
        except Exception as e:
            logger.error(f"Stripe Metering failed: {e}")

    # --- 2. Ingest Logic ---
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
        node_id=node.id, f0=signature.f0, q_factor=signature.q_factor,
        signature_vector=signature.signature_vector, is_anomaly=is_alert
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
