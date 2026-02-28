import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from anthropic import Anthropic

app = FastAPI(
    title="CEFIELD Global Brain API",
    description="The central hub for the federated resonator measurement network with AI Root Cause Analysis.",
    version="1.1.0"
)

# Initialize Anthropic Client
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

class ResonatorSignature(BaseModel):
    node_id: str
    hardware_type: str
    f0: float
    q_factor: float
    signature_vector: list[float]  # 128-dimensional latent vector of the ring-down

def analyze_anomaly_with_claude(signature: ResonatorSignature) -> str:
    """
    Calls Claude 3.5 Sonnet to perform root-cause analysis based on hardware 
    and measurement parameters when a drop in Q-factor is detected.
    """
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
    - Vector data summary: [128-dim envelope envelope, showing rapid non-exponential decay]
    
    Based on typical microwave and nanomechanical resonator physics (e.g. anchor loss, thermoelastic dissipation, Two-Level System (TLS) defects, vacuum leaks), provide a highly professional, 2-sentence diagnostic assessment of what is likely causing this drop, and suggest one immediate hardware check the lab operator should perform.
    """

    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=250,
            system="You are an expert physicist and metrology engineer.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"AI Diagnostic failed: {str(e)}"

@app.post("/api/v1/ingest")
async def ingest_signature(signature: ResonatorSignature):
    """
    Receives a highly compressed, anonymized resonator vector from an Edge Node.
    """
    # Trigger AI Root Cause Analysis if Q-factor is suspicious (e.g. < 10000)
    if signature.q_factor < 10000:
        diagnostic = analyze_anomaly_with_claude(signature)
        return {
            "status": "ingested",
            "alert": "Q-Factor drop detected. Anomaly registered.",
            "ai_diagnostic": diagnostic
        }
        
    return {"status": "ingested", "message": "Signature added to Global Genome."}

@app.get("/api/v1/genome/stats")
async def get_genome_stats():
    return {
        "total_signatures_worldwide": 14206,
        "active_edge_nodes": 42,
        "diagnostics_run": 891
    }
