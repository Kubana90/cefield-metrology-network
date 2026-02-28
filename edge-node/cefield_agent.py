import time
import json
import httpx
import numpy as np
from scipy.signal import hilbert

CLOUD_API_URL = "http://cloud-core:8000/api/v1/ingest"
NODE_ID = "lab-munich-quantum-01"

def acquire_raw_signal():
    """
    Mock function representing hardware interface (e.g., Red Pitaya SCPI).
    Returns raw RF ring-down data.
    """
    t = np.linspace(0, 1e-3, 10000)
    decay = np.exp(-t / 1e-4)
    carrier = np.sin(2 * np.pi * 1.5e9 * t)
    noise = np.random.normal(0, 0.05, len(t))
    return t, (decay * carrier) + noise

def extract_signature_vector(signal):
    """
    Compresses raw data into a latent vector (fingerprint) for the Swarm DB.
    Instead of sending gigabytes of RF data, we send 128 floats.
    """
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    
    indices = np.linspace(0, len(amplitude_envelope)-1, 128, dtype=int)
    signature = amplitude_envelope[indices].tolist()
    
    return signature

def main():
    print(f"[{NODE_ID}] CEFIELD Edge Agent started.")
    time.sleep(5) # Wait for API to boot
    while True:
        print("Acquiring hardware measurement...")
        t, signal = acquire_raw_signal()
        
        print("Extracting physical signature...")
        vector = extract_signature_vector(signal)
        
        payload = {
            "node_id": NODE_ID,
            "hardware_type": "Red_Pitaya_STEMlab_125-14",
            "f0": 1.5e9,
            "q_factor": 8500,  # Simulated drop
            "signature_vector": vector
        }
        
        print("Streaming to CEFIELD Global Brain...")
        try:
            response = httpx.post(CLOUD_API_URL, json=payload)
            print(f"Cloud Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Connection failed: {e}")
            
        time.sleep(5) # Measure every 5 seconds

if __name__ == "__main__":
    main()
