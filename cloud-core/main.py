import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.responses import HTMLResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from database import engine, get_db, Base
from models import Node, Measurement
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cefield")

# Initialize DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="CEFIELD Global Brain API",
    description="Federated resonator network with AI Root Cause Analysis & pgvector Similarity Search.",
    version="1.4.0"
)

# -----------------------------
# Auth (API Keys)
# -----------------------------
API_KEY_HEADER = APIKeyHeader(name="X-CEFIELD-API-KEY", auto_error=True)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
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


def find_similar_anomalies(db: Session, query_vector: list[float], limit: int = 3) -> List[Tuple[Measurement, float, str]]:
    """
    Uses pgvector L2 distance (<->) to find the most mathematically similar 
    resonator signatures across the global database.
    """
    # Find similar vectors where q_factor was also low (historical anomalies)
    # L2 distance: Measurement.signature_vector.l2_distance(query_vector)
    # We join with Node to get the lab names
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
        # Convert distance to a rough similarity percentage (heuristic for demo)
        similarity_score = max(0.0, 100.0 - (dist * 100.0)) 
        similar_cases.append((meas, similarity_score, lab_name))
        
    return similar_cases


def analyze_anomaly_with_claude(signature: ResonatorSignature, similar_cases: List[Tuple[Measurement, float, str]]) -> str:
    """Calls Claude with current hardware data PLUS historical swarm intelligence."""
    if not anthropic_client:
        return "AI Diagnostic disabled. Missing ANTHROPIC_API_KEY."

    # Build context from the global swarm
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

Based on typical resonator physics and the historical swarm matches provided, provide a highly professional, 2-sentence diagnostic assessment of what is likely causing this drop. Make sure to reference the historical matches if they exist to demonstrate swarm intelligence. Suggest one immediate hardware check.
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
    html_path = Path(__file__).parent / "web" / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

@app.get("/api/v1/nodes")
async def list_nodes(db: Session = Depends(get_db)):
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
    # 1. Upsert Node
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

    if signature.lab_name: node.lab_name = signature.lab_name
    if signature.lat is not None: node.lat = signature.lat
    if signature.lon is not None: node.lon = signature.lon

    # 2. Similarity Search BEFORE saving the new one (so we don't just match against ourselves)
    similar_cases = []
    if is_alert:
        similar_cases = find_similar_anomalies(db, signature.signature_vector, limit=3)

    # 3. Store the actual measurement
    measurement = Measurement(
        node_id=node.id,
        f0=signature.f0,
        q_factor=signature.q_factor,
        signature_vector=signature.signature_vector,
        is_anomaly=is_alert
    )
    db.add(measurement)
    db.commit()

    # 4. AI Analysis with Swarm Context
    if is_alert:
        diagnostic = analyze_anomaly_with_claude(signature, similar_cases)
        return {
            "status": "ingested",
            "alert": "Q-Factor drop detected. Anomaly registered.",
            "swarm_matches": [
                {"similarity_score": round(score, 1), "matched_lab": lab} 
                for _, score, lab in similar_cases
            ],
            "ai_diagnostic": diagnostic,
        }

    return {"status": "ingested", "message": f"Measurement committed for {signature.node_id}."}
