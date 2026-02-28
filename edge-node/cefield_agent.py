import time
import json
import httpx
import numpy as np
from scipy.signal import hilbert

CLOUD_API_URL = "http://cloud-core:8000/api/v1/ingest"
API_KEY = "cef_dev_machine_001"

# We simulate two different labs to show similarity matching over time
NODES = [
    {
        "node_id": "lab-stanford-mems-01",
        "lab_name": "Stanford MEMS Foundry",
        "hardware_type": "NI_PXIe_5170R",
        "lat": 37.4275,
        "lon": -122.1697
    },
    {
        "node_id": "lab-munich-quantum-01",
        "lab_name": "Munich Quantum Lab",
        "hardware_type": "Red_Pitaya_STEMlab_125-14",
        "lat": 48.1351,
        "lon": 11.5820
    }
]

def acquire_raw_signal(is_stanford=False):
    t = np.linspace(0, 1e-3, 10000)
    
    # We generate a specific "TLS defect" signature: 
    # a rapid decay early on, followed by a slower tail
    if is_stanford:
        decay = np.exp(-t / 5e-5) + 0.2 * np.exp(-t / 3e-4) 
    else:
        # Munich has a VERY similar defect, just slightly different noise
        decay = np.exp(-t / 5.2e-5) + 0.18 * np.exp(-t / 3.1e-4)
        
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
    print(f"[*] CEFIELD Swarm Simulation Agent started.")
    time.sleep(5)
    
    headers = {
        "X-CEFIELD-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }
    
    # Step 1: Stanford measures a defect and stores it in the Global Genome
    print("\n--- STEP 1: Stanford measures a defect ---")
    t, signal = acquire_raw_signal(is_stanford=True)
    vector = extract_signature_vector(signal)
    
    payload_stanford = NODES[0].copy()
    payload_stanford.update({
        "f0": 1.5e9,
        "q_factor": 4200, # Big drop
        "signature_vector": vector
    })
    
    try:
        response = httpx.post(CLOUD_API_URL, json=payload_stanford, headers=headers)
        print(f"Stanford Upload: {response.status_code}")
    except Exception as e:
        print(f"Connection failed: {e}")
        
    time.sleep(3)
    
    # Step 2: Munich measures the exact same physical phenomenon
    print("\n--- STEP 2: Munich measures a similar defect. Triggering Swarm Match! ---")
    while True:
        t, signal = acquire_raw_signal(is_stanford=False)
        vector = extract_signature_vector(signal)

        payload_munich = NODES[1].copy()
        payload_munich.update({
            "f0": 1.5e9,
            "q_factor": 4500,
            "signature_vector": vector
        })

        try:
            response = httpx.post(CLOUD_API_URL, json=payload_munich, headers=headers)
            res_json = response.json()
            print(f"\n[MUNICH CLOUD RESPONSE]")
            print(f"Alert: {res_json.get('alert')}")
            
            if 'swarm_matches' in res_json:
                for match in res_json['swarm_matches']:
                    print(f"-> SWARM MATCH: {match['similarity_score']}% match with {match['matched_lab']}")
                    
            print(f"\nClaude Diagnostic: {res_json.get('ai_diagnostic')}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Connection failed: {e}")

        time.sleep(10)

if __name__ == "__main__":
    main()
