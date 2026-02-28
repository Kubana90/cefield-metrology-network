import httpx
from typing import Optional, Dict, Any
import numpy as np
from .processor import SignalProcessor

class CefieldNode:
    """
    The main client class to interface with the CEFIELD Cloud Core.
    """
    def __init__(self, node_id: str, api_url: str = "http://localhost:8000"):
        self.node_id = node_id
        self.api_url = api_url.rstrip("/")
        self.client = httpx.Client(timeout=10.0)

    def analyze_and_stream(self, time_array: np.ndarray, voltage_array: np.ndarray, hardware_type: str, estimated_f0: float) -> Dict[str, Any]:
        """
        Locally processes the raw array, extracts the signature, and streams it to the Global Brain.
        """
        # 1. Edge Compute (Local)
        signature_vector = SignalProcessor.extract_signature(time_array, voltage_array)
        q_factor = SignalProcessor.estimate_q_factor(time_array, voltage_array, estimated_f0)
        
        # 2. Prepare payload
        payload = {
            "node_id": self.node_id,
            "hardware_type": hardware_type,
            "f0": estimated_f0,
            "q_factor": q_factor,
            "signature_vector": signature_vector
        }
        
        # 3. Stream to Cloud Core
        try:
            response = self.client.post(f"{self.api_url}/api/v1/ingest", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"status": "error", "message": f"Cloud API error: {e.response.text}"}
        except Exception as e:
            return {"status": "error", "message": f"Connection failed: {str(e)}"}
