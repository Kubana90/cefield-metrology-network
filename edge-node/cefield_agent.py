import time
import json
import httpx
import numpy as np
from scipy.signal import hilbert

CLOUD_API_URL = "http://cloud-core:8000/api/v1/ingest"
NODE_ID = "lab-munich-quantum-01"
API_KEY = "cef_dev_machine_001" # Simulating an authenticated device

def acquire_raw_signal():
    t = np.linspace(0, 1e-3, 10000)
    decay = np.exp(-t / 1e-4)
    carrier = np.sin(2 * np.pi * 1.5e9 * t)
    noise = np.random.normal(0, 0.05, len(t))
    return t, (decay * carrier) + noise

def extract_signature_vector(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    indices = np.linspace(0, len(amplitude_envelope)-1, 128, dtype=int)
    signature = amplitude_envelope[indices].tolist()
    return signature

def main():
    print(f"[{NODE_ID}] CEFIELD Authenticated Edge Agent started.")
    time.sleep(5)
    
    headers = {
        "X-CEFIELD-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }
    
    while True:
        t, signal = acquire_raw_signal()
        vector = extract_signature_vector(signal)

        payload = {
            "node_id": NODE_ID,
            "hardware_type": "Red_Pitaya_STEMlab_125-14",
            "f0": 1.5e9,
            "q_factor": 8500,
            "signature_vector": vector,
            "lab_name": "Munich Quantum Lab (Sim)",
            "lat": 48.1351,
            "lon": 11.5820
        }

        try:
            response = httpx.post(CLOUD_API_URL, json=payload, headers=headers)
            print(f"Cloud Response ({response.status_code}): {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Connection failed: {e}")

        time.sleep(5)

if __name__ == "__main__":
    main()
