import os
import logging
import stripe
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from database import engine, get_db, Base, Organization, Node, Measurement
from models import Node as OldNode, Measurement as OldMeasurement # Keeping imports clean
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cefield")

# Initialize DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="CEFIELD Global Brain API - Enterprise",
    description="Federated resonator network with pgvector Similarity Search & Stripe B2B Billing.",
    version="2.0.0"
)

# -----------------------------
# Config & Keys
# -----------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# Stripe Setup
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "sk_test_mock_123")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "whsec_mock_123")
stripe.api_key = STRIPE_API_KEY

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

# -----------------------------
# Auth (API Keys mapped to DB)
# -----------------------------
API_KEY_HEADER = APIKeyHeader(name="X-CEFIELD-API-KEY", auto_error=True)

def get_organization_from_api_key(api_key: str = Security(API_KEY_HEADER), db: Session = Depends(get_db)) -> Organization:
    """Validates the API key and returns the paying Organization. Blocks if subscription is dead."""
    # Hardcoded bypass for the simulation script (Munich/Stanford demo)
    if api_key == "cef_dev_machine_001":
        # Ensure a mock org exists for the demo
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
# Stripe Billing Endpoints
# -----------------------------

@app.post("/api/v1/billing/onboard")
async def onboard_customer(data: CustomerCreate, db: Session = Depends(get_db)):
    """
    Creates a new B2B customer in Stripe and generates an API key for them.
    In a real app, this returns a Stripe Checkout URL to enter credit card details.
    """
    try:
        # Create Stripe Customer
        # In a real scenario, you'd use stripe.Customer.create()
        mock_stripe_id = f"cus_{int(utc_now().timestamp())}"
        
        # Generate a secure API Key
        import secrets
        new_api_key = f"cef_{secrets.token_hex(16)}"
        
        org = Organization(
            name=data.org_name,
            api_key=new_api_key,
            stripe_customer_id=mock_stripe_id,
            subscription_active=True,  # Usually set to True via webhook after payment
            subscription_tier="enterprise",
            created_at=utc_now()
        )
        db.add(org)
        db.commit()
        
        return {
            "message": "Organization created. Please save your API Key securely.",
            "api_key": new_api_key,
            "stripe_customer_id": mock_stripe_id,
            "billing_portal_url": "https://billing.stripe.com/p/session/mock_url_for_demo"
        }
    except Exception as e:
        logger.error(f"Onboarding failed: {e}")
        raise HTTPException(status_code=500, detail="Could not create customer billing profile.")


@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Listens for Stripe events (failed payments, upgrades, cancellations)."""
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        # verify webhook signature (mocked here for demo, use real stripe sdk method)
        # event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        event = await request.json() # simplified for demo
    except Exception as e:
        raise HTTPException(status_code=400, detail="Webhook Error")

    event_type = event.get("type")
    
    # Extract customer ID from the event object
    data_object = event.get("data", {}).get("object", {})
    customer_id = data_object.get("customer")

    if not customer_id:
        return JSONResponse({"status": "ignored"})

    org = db.query(Organization).filter(Organization.stripe_customer_id == customer_id).first()
    if not org:
        return JSONResponse({"status": "customer not found"})

    if event_type == "invoice.payment_failed":
        logger.warning(f"Payment failed for {org.name}. Deactivating API access.")
        org.subscription_active = False
        db.commit()
        
    elif event_type == "invoice.payment_succeeded":
        logger.info(f"Payment succeeded for {org.name}. Activating API access.")
        org.subscription_active = True
        db.commit()

    elif event_type == "customer.subscription.deleted":
        logger.warning(f"Subscription cancelled for {org.name}.")
        org.subscription_active = False
        org.subscription_tier = "freemium"
        db.commit()

    return JSONResponse({"status": "success"})


# -----------------------------
# Core Diagnostic logic (from previous step)
# -----------------------------

def find_similar_anomalies(db: Session, query_vector: list[float], limit: int = 3) -> List[Tuple[Measurement, float, str]]:
    results = (
        db.query(Measurement, Node.lab_name, Measurement.signature_vector.l2_distance(query_vector).label("distance"))
        .join(Node)
        .filter(Measurement.is_anomaly == True)
        .order_by("distance")
        .limit(limit)
        .all()
    )
    similar_cases = []
    for meas, lab_name, dist in results:
        similarity_score = max(0.0, 100.0 - (dist * 100.0)) 
        similar_cases.append((meas, similarity_score, lab_name))
    return similar_cases


def analyze_anomaly_with_claude(signature: ResonatorSignature, similar_cases: List[Tuple[Measurement, float, str]]) -> str:
    if not anthropic_client:
        return "AI Diagnostic disabled. Missing ANTHROPIC_API_KEY."

    swarm_context = "Historical matches from the Global Resonator Genome:\n"
    if not similar_cases:
        swarm_context += "- No exact historical matches found yet. This is a novel anomaly pattern.\n"
    else:
        for meas, score, lab in similar_cases:
            lab_display = lab or "Unknown Lab"
            swarm_context += f"- MATCH: {score:.1f}% similarity to an anomaly recorded at {lab_display} (f0: {meas.f0}Hz, Q: {meas.q_factor}).\n"

    prompt = f"""
You are the CEFIELD Root Cause Diagnostic AI.
A laboratory node reported an anomalous measurement:
- Node ID: {signature.node_id}
- Hardware: {signature.hardware_type}
- Frequency (f0): {signature.f0} Hz
- Q-Factor: {signature.q_factor} (Anomalous drop detected)

{swarm_context}

Based on typical resonator physics and the historical swarm matches provided, provide a highly professional, 2-sentence diagnostic assessment of what is likely causing this drop.
""".strip()

    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=250,
            system="You are an expert physicist and metrology engineer.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Claude API failed: {e}")
        return f"AI Diagnostic failed: {str(e)}"


# -----------------------------
# Data Ingest & Metering
# -----------------------------

@app.post("/api/v1/ingest")
async def ingest_signature(
    signature: ResonatorSignature, 
    org: Organization = Depends(get_organization_from_api_key),
    db: Session = Depends(get_db)
):
    # --- 1. METERING FOR BILLING ---
    # In a production system, we log a usage event to Stripe to charge them $0.05 per ingest
    if org.stripe_customer_id and org.subscription_tier == "enterprise":
        try:
            # stripe.billing.MeterEvent.create(
            #     event_name="vector_upload",
            #     payload={"value": "1", "stripe_customer_id": org.stripe_customer_id}
            # )
            logger.info(f"Billing: Billed 1 unit to {org.name}")
        except Exception as e:
            logger.error(f"Failed to report usage to Stripe: {e}")

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

    if signature.lab_name: node.lab_name = signature.lab_name
    if signature.lat is not None: node.lat = signature.lat
    if signature.lon is not None: node.lon = signature.lon

    similar_cases = []
    if is_alert:
        similar_cases = find_similar_anomalies(db, signature.signature_vector, limit=3)

    measurement = Measurement(
        node_id=node.id,
        f0=signature.f0,
        q_factor=signature.q_factor,
        signature_vector=signature.signature_vector,
        is_anomaly=is_alert
    )
    db.add(measurement)
    db.commit()

    if is_alert:
        diagnostic = analyze_anomaly_with_claude(signature, similar_cases)
        return {
            "status": "ingested",
            "organization": org.name,
            "alert": "Q-Factor drop detected. Anomaly registered.",
            "swarm_matches": [
                {"similarity_score": round(score, 1), "matched_lab": lab} 
                for _, score, lab in similar_cases
            ],
            "ai_diagnostic": diagnostic,
        }

    return {"status": "ingested", "message": f"Measurement committed for {signature.node_id} (Org: {org.name})."}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "web" / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
