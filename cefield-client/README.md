# CEFIELD Python Client (`cefield-client`)

The official command-line interface (CLI) and Python SDK for connecting your laboratory hardware to the CEFIELD Global Resonator Genome.

## Features
- **Hardware Agnostic**: Read data from CSV, standard numpy arrays, or directly from PicoScope/RedPitaya APIs.
- **Local Processing**: Performs heavy FFT and Hilbert transforms locally.
- **Privacy First**: Extracts only the 128-dimensional physical signature (latent vector). Raw proprietary data never leaves your lab.
- **Instant Diagnostics**: Receive real-time Root Cause AI analysis if your measurement deviates from the global baseline.

## Installation

```bash
pip install cefield-client
```
*(Currently installed from source in this repository)*

## Quickstart (CLI)

Analyze a local CSV file containing `time` and `voltage` columns:

```bash
cefield analyze my_resonator_data.csv --f0 1.5e9 --hardware "PicoScope_6000" --node "Lab_Munich_01"
```

## Python API Usage

```python
import numpy as np
from cefield import CefieldNode

# 1. Initialize your Edge Node
node = CefieldNode(
    api_key="your_cefield_api_key",
    node_id="quantum-lab-berlin"
)

# 2. Acquire your data (e.g., from your ADC)
t = np.linspace(0, 1e-3, 10000)
signal = np.exp(-t / 1e-4) * np.sin(2 * np.pi * 1.5e9 * t)

# 3. Stream to the Global Brain & Get AI Diagnostics
response = node.analyze_and_stream(
    time_array=t,
    voltage_array=signal,
    hardware_type="Custom_FPGA",
    estimated_f0=1.5e9
)

print(response.ai_diagnostic)
```
