<div align="center">
  <img src="https://raw.githubusercontent.com/Kubana90/cefield-metrology-network/main/docs/assets/cefield-logo-placeholder.png" alt="CEFIELD Logo" width="200" onerror="this.src='https://via.placeholder.com/200x60/0b1220/38bdf8?text=CEFIELD'"/>
  
  # CEFIELD: The Global Resonator Genome
  
  **The world's first decentralized, LLM-driven measurement network for the quantum & semiconductor industry.**

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
  [![PostgreSQL pgvector](https://img.shields.io/badge/pgvector-Supported-336791.svg?logo=postgresql)](https://github.com/pgvector/pgvector)
  [![Claude 3.5 Sonnet](https://img.shields.io/badge/AI_Engine-Claude_3.5_Sonnet-D97757.svg)](https://www.anthropic.com/claude)
</div>

---

## 🚀 The Vision: "GitHub for Physical Metrology"

Currently, highly valuable hardware measurement data (like resonator ring-downs in quantum computing or MEMS manufacturing) decays in isolated silos on local lab hard drives. A €50,000 oscilloscope in Munich learns absolutely nothing from a measurement taken in Stanford.

**CEFIELD breaks this paradigm.**

We decouple the intelligence from expensive hardware. By using affordable edge ADCs (like Red Pitaya or SDRs) running the `cefield-client`, we compress raw RF signals into lightweight, privacy-preserving **latent vectors**. These vectors are streamed to the CEFIELD Global Brain, creating a worldwide swarm-intelligence database.

When an anomaly occurs, our system performs a **millisecond vector-similarity search (pgvector)** across all global labs, and feeds the historical context to **Claude 3.5 Sonnet** to deliver an instant, physics-informed Root Cause Analysis.

---

## 🧠 Core Features

1. **Hardware-Agnostic Edge Compute**: Runs locally on your lab laptop or directly on FPGAs. It performs heavy DSP (Hilbert transforms, FFT) locally. Raw proprietary data *never* leaves your lab.
2. **Latent Vector Extraction**: We compress megabytes of time-series RF data into a `128-dimensional` physical signature vector.
3. **Swarm Intelligence (pgvector)**: Global $L_2$ distance matching. If your quantum chip fails, CEFIELD tells you if Stanford saw the exact same failure mode 3 weeks ago.
4. **Autonomous AI Diagnostics**: Claude 3.5 Sonnet acts as an expert metrology engineer, interpreting the vector data and swarm context to give you actionable hardware advice.
5. **Real-Time Admin Console**: A beautiful, dark-mode world map tracking all active nodes and hardware health globally.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph "Lab Edge (Local Compute)"
        HW[Lab Hardware / ADC] --> |Raw RF Data| Agent[CEFIELD Python Agent]
        Agent --> |Extracts 128-dim Vector| Agent
    end
    
    subgraph "CEFIELD Global Brain (Cloud)"
        API[FastAPI Gateway]
        DB[(PostgreSQL + pgvector)]
        LLM[Claude 3.5 Sonnet]
        
        Agent -->|Authenticates via API Key| API
        API <-->|Stores & Matches Vector| DB
        API <-->|Root Cause Analysis| LLM
    end
```

---

## 💻 Quickstart (Simulation Mode)

Want to see the swarm intelligence in action? You can spin up the entire cloud architecture and simulate two global labs (Stanford and Munich) interacting with each other.

### 1. Prerequisites
- Docker and Docker Compose installed.
- An Anthropic API Key (for Claude 3.5 Sonnet).

### 2. Setup
Clone the repository and add your API key:
```bash
git clone https://github.com/Kubana90/cefield-metrology-network.git
cd cefield-metrology-network

# Create an environment file
echo "ANTHROPIC_API_KEY=sk-ant-api03-your-real-key-here" > .env
```

### 3. Launch the Network
```bash
docker-compose up --build
```

### 4. Watch the Magic Happen
In your terminal, you will see the simulation unfold:
1. **Stanford Lab** measures a defective chip and uploads the vector to the Global Genome.
2. **Munich Lab** measures a chip with a similar defect.
3. The Cloud API detects the anomaly, finds the Stanford match via `pgvector`, and Claude provides a diagnosis linking the two!

Open the **Admin Console** in your browser to see the live map:
👉 `http://localhost:8000/dashboard`

---

## 📦 Using the Python CLI (`cefield-client`)

If you want to stream real hardware data from your own CSV files:

```bash
cd cefield-client
pip install -e .

# Analyze a local dataset and stream to the Global Brain
cefield analyze my_data.csv --f0 1.5e9 --hardware "PicoScope_6000" --node "My_Lab_01"
```

---

## 🛡️ Privacy & Security
We understand that DeepTech foundries cannot share proprietary RF data. 
- **No Raw Data:** We only transmit a normalized, 128-dimensional envelope shape. 
- **Opt-in Metadata:** Geo-location and Lab Names are entirely optional.

---
*Built by Operator-996. Designed to break the system and reach a new level of production-grade innovation.*
