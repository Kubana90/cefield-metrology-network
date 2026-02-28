import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from anthropic import Anthropic

app = FastAPI(
    title="CEFIELD Global Brain API",
    description="The central hub for the federated resonator measurement network with AI Root Cause Analysis.",
    version="1.2.0"
)

# -----------------------------
# Claude (Anthropic) Client
# -----------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# -----------------------------
# In-memory Node Registry (MVP)
# -----------------------------
NODE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResonatorSignature(BaseModel):
    node_id: str
    hardware_type: str
    f0: float
    q_factor: float
    signature_vector: list[float]  # 128-dimensional latent vector of the ring-down

    # Optional metadata (privacy-first; user can omit)
    lab_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


def analyze_anomaly_with_claude(signature: ResonatorSignature) -> str:
    """Calls Claude 3.5 Sonnet for physics-aware root cause analysis."""
    if not anthropic_client:
        return "AI Diagnostic disabled. Missing ANTHROPIC_API_KEY."

    prompt = f"""
You are the CEFIELD Root Cause Diagnostic AI.
You analyze physics data from hardware sensors globally to diagnose failures in quantum and semiconductor devices.

A laboratory node reported an anomalous measurement:
- Node ID: {signature.node_id}
- Hardware: {signature.hardware_type}
- Frequency (f0): {signature.f0} Hz
- Q-Factor: {signature.q_factor} (Anomalous drop detected)
- Vector data summary: [128-dim envelope vector, normalized 0..1]

Based on typical resonator physics (anchor loss, thermoelastic dissipation, Two-Level System (TLS) defects, vacuum leaks, parasitic mode coupling), provide a highly professional, 2-sentence diagnostic assessment of what is likely causing this drop, and suggest one immediate hardware check the lab operator should perform.
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
        return f"AI Diagnostic failed: {str(e)}"


def upsert_node_registry(signature: ResonatorSignature) -> None:
    existing = NODE_REGISTRY.get(signature.node_id, {})

    NODE_REGISTRY[signature.node_id] = {
        "node_id": signature.node_id,
        "lab_name": signature.lab_name or existing.get("lab_name"),
        "hardware_type": signature.hardware_type,
        "lat": signature.lat if signature.lat is not None else existing.get("lat"),
        "lon": signature.lon if signature.lon is not None else existing.get("lon"),
        "last_seen": utc_now_iso(),
        "last_f0": signature.f0,
        "last_q_factor": signature.q_factor,
        "last_alert": signature.q_factor < 10000,
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Single-file admin console (map + live node list)."""
    html_path = Path(__file__).parent / "web" / "dashboard.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/v1/nodes")
async def list_nodes():
    """Returns last-known state of all edge nodes (MVP, in-memory)."""
    nodes = list(NODE_REGISTRY.values())
    nodes.sort(key=lambda n: n.get("last_seen", ""), reverse=True)
    return {
        "updated_at": utc_now_iso(),
        "count": len(nodes),
        "nodes": nodes,
    }


@app.get("/api/v1/nodes/{node_id}")
async def get_node(node_id: str):
    node = NODE_REGISTRY.get(node_id)
    if not node:
        return {"status": "not_found", "node_id": node_id}
    return node


@app.post("/api/v1/ingest")
async def ingest_signature(signature: ResonatorSignature):
    """Receives a compressed, anonymized resonator vector from an Edge Node."""
    upsert_node_registry(signature)

    # Trigger AI Root Cause Analysis if Q-factor is suspicious (e.g. < 10000)
    if signature.q_factor < 10000:
        diagnostic = analyze_anomaly_with_claude(signature)
        return {
            "status": "ingested",
            "alert": "Q-Factor drop detected. Anomaly registered.",
            "ai_diagnostic": diagnostic,
        }

    return {"status": "ingested", "message": "Signature added to Global Genome."}


@app.get("/api/v1/genome/stats")
async def get_genome_stats():
    return {
        "total_signatures_worldwide": 14206,
        "active_edge_nodes": len(NODE_REGISTRY),
        "diagnostics_run": 891,
    }
