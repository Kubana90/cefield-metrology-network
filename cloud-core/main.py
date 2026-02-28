import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import engine, get_db, Base
from models import Node, Measurement
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cefield")

# Initialize DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="CEFIELD Global Brain API",
    description="The central hub for the federated resonator measurement network.",
    version="1.3.0"
)

# -----------------------------
# Auth (API Keys)
# -----------------------------
API_KEY_HEADER = APIKeyHeader(name="X-CEFIELD-API-KEY", auto_error=True)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """
    In production, this would check against an 'organizations' table.
    For this MVP, we allow a hardcoded demo key or any key starting with 'cef_'.
    """
    if api_key == "demo_key_123" or api_key.startswith("cef_"):
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")


# -----------------------------
# Claude (Anthropic) Client
# -----------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

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


def analyze_anomaly_with_claude(signature: ResonatorSignature) -> str:
    """Calls Claude 3.5 Sonnet for physics-aware root cause analysis."""
    if not anthropic_client:
        return "AI Diagnostic disabled. Missing ANTHROPIC_API_KEY."

    prompt = f"""
You are the CEFIELD Root Cause Diagnostic AI.
A laboratory node reported an anomalous measurement:
- Node ID: {signature.node_id}
- Hardware: {signature.hardware_type}
- Frequency (f0): {signature.f0} Hz
- Q-Factor: {signature.q_factor} (Anomalous drop detected)

Based on typical resonator physics, provide a highly professional, 2-sentence diagnostic assessment of what is likely causing this drop, and suggest one immediate hardware check.
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
# API Endpoints
# -----------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Single-file admin console (map + live node list)."""
    html_path = Path(__file__).parent / "web" / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/v1/nodes")
async def list_nodes(db: Session = Depends(get_db)):
    """Returns state of all registered edge nodes from PostgreSQL."""
    nodes = db.query(Node).order_by(Node.last_seen.desc()).all()
    return {
        "updated_at": utc_now().isoformat(),
        "count": len(nodes),
        "nodes": [
            {
                "node_id": n.node_id,
                "lab_name": n.lab_name,
                "hardware_type": n.hardware_type,
                "lat": n.lat,
                "lon": n.lon,
                "last_seen": n.last_seen.isoformat() if n.last_seen else None,
                "last_q_factor": n.last_q_factor,
                "last_alert": n.last_alert
            } for n in nodes
        ],
    }

@app.post("/api/v1/ingest")
async def ingest_signature(
    signature: ResonatorSignature, 
    api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Receives a compressed vector from an Authenticated Edge Node.
    Stores the measurement in Postgres and updates Node status.
    """
    # 1. Upsert Node (Enrollment on first contact)
    node = db.query(Node).filter(Node.node_id == signature.node_id).first()
    if not node:
        logger.info(f"Registering new node: {signature.node_id}")
        node = Node(node_id=signature.node_id)
        db.add(node)
    
    node.hardware_type = signature.hardware_type
    node.last_seen = utc_now()
    node.last_q_factor = signature.q_factor
    
    is_alert = signature.q_factor < 10000
    node.last_alert = is_alert

    # Only update static metadata if provided
    if signature.lab_name: node.lab_name = signature.lab_name
    if signature.lat is not None: node.lat = signature.lat
    if signature.lon is not None: node.lon = signature.lon

    # 2. Store the actual measurement (Commit)
    # Note: signature_vector is stored as pgvector embedding
    measurement = Measurement(
        node_id=node.id,
        f0=signature.f0,
        q_factor=signature.q_factor,
        signature_vector=signature.signature_vector,
        is_anomaly=is_alert
    )
    db.add(measurement)
    db.commit()

    # 3. AI Analysis if anomaly
    if is_alert:
        diagnostic = analyze_anomaly_with_claude(signature)
        # We could save the diagnostic back to the DB here
        return {
            "status": "ingested",
            "alert": "Q-Factor drop detected. Anomaly registered.",
            "ai_diagnostic": diagnostic,
        }

    return {"status": "ingested", "message": f"Measurement committed for {signature.node_id}."}

@app.get("/api/v1/genome/stats")
async def get_genome_stats(db: Session = Depends(get_db)):
    node_count = db.query(Node).count()
    measurement_count = db.query(Measurement).count()
    return {
        "total_signatures_worldwide": measurement_count,
        "active_edge_nodes": node_count,
        "status": "online"
    }
