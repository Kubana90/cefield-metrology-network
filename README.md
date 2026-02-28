# CEFIELD Metrology Network 🌐

> The "GitHub for Physical Metrology"

Welcome to the **CEFIELD Global Resonator Genome**. This repository contains the foundational architecture for the world's first decentralized, LLM-driven measurement network for the quantum and semiconductor industry.

## 🏗️ Architecture

1. **Edge Node (`/edge-node`)**: A lightweight Python agent running directly on ADCs (Red Pitaya, SDRs). Extracts resonator signatures (vectors) and streams them to the cloud.
2. **Cloud Core (`/cloud-core`)**: FastAPI brain using PostgreSQL with `pgvector` to store and match physical resonance signatures. Orchestrated by Claude AI for Root Cause Analysis.

## 🚀 Getting Started

Simply spin up the simulation environment with Docker Compose. This starts the Vector DB, the Cloud API, and simulates an Edge Node taking measurements.

```bash
docker-compose up --build
```
