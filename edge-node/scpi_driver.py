"""
CEFIELD Edge SCPI Driver
=========================
Integrates the SCPI bridge with the edge-node DSP pipeline.

This module replaces the simulated signal generation in cefield_agent.py
with real instrument data acquisition when SCPI hardware is available.

Pipeline:
  SCPI Instrument → Raw Waveform → Hilbert DSP → 128-dim Vector → CEFIELD API

Usage:
    from scpi_driver import SCPIEdgeDriver

    driver = SCPIEdgeDriver.from_env()
    if driver and driver.connect():
        signal = driver.acquire()
        vector = driver.extract_signature(signal)
        # → send vector to CEFIELD cloud-core

Fallback:
    If no SCPI hardware is configured, the edge-agent gracefully
    falls back to simulation mode (existing cefield_agent.py behavior).
"""
from __future__ import annotations

import os
import socket
import logging
import time
from typing import Optional
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

logger = logging.getLogger("cefield.edge.scpi")


@dataclass
class SCPIConfig:
    """Edge-side SCPI configuration."""
    host: str
    port: int = 5000
    hardware_type: str = "generic"
    timeout: float = 5.0
    buffer_size: int = 65536
    node_id: str = "scpi-edge-01"
    lab_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    f0_estimate_hz: float = 1.5e9


# Device-specific command mappings for edge-side acquisition
ACQUIRE_COMMANDS: dict[str, dict[str, str]] = {
    "red_pitaya": {
        "idn": "*IDN?",
        "reset": "*RST",
        "format": "ACQ:DATA:FORMAT ASCII",
        "decimate": "ACQ:DEC 1",
        "trigger": "ACQ:TRIG NOW",
        "start": "ACQ:START",
        "status": "ACQ:TRIG:STAT?",
        "data": "ACQ:SOUR1:DATA?",
        "stop": "ACQ:STOP",
    },
    "keysight": {
        "idn": "*IDN?",
        "reset": "*RST",
        "format": ":WAVEFORM:FORMAT ASCII",
        "decimate": ":TIMEBASE:RANGE 1E-3",
        "trigger": ":TRIGGER:EDGE:SOURCE CHAN1",
        "start": ":DIGITIZE CHAN1",
        "status": ":OPER:COND?",
        "data": ":WAVEFORM:DATA?",
        "stop": ":STOP",
    },
    "picoscope": {
        "idn": "*IDN?",
        "reset": "*RST",
        "format": ":FORMAT:DATA ASCii",
        "decimate": ":TIMEBASE:RANGE 1E-3",
        "trigger": ":TRIGGER:MAIN:SOURCE CH1",
        "start": ":RUN",
        "status": ":STATUS?",
        "data": ":FETCH:WAV?",
        "stop": ":STOP",
    },
    "rohde_schwarz": {
        "idn": "*IDN?",
        "reset": "*RST",
        "format": "FORM ASC",
        "decimate": "TIM:RANG 1E-3",
        "trigger": "TRIG:SOUR CH1",
        "start": "RUN",
        "status": "ACQ:STAT?",
        "data": "CHAN1:DATA?",
        "stop": "STOP",
    },
    "generic": {
        "idn": "*IDN?",
        "reset": "*RST",
        "format": "ACQ:DATA:FORMAT ASCII",
        "decimate": "ACQ:DEC 1",
        "trigger": "ACQ:TRIG NOW",
        "start": "ACQ:START",
        "status": "ACQ:STATUS?",
        "data": "ACQ:SOUR1:DATA?",
        "stop": "ACQ:STOP",
    },
}


class SCPIEdgeDriver:
    """
    Edge-side SCPI driver that acquires real waveforms and feeds
    them into the CEFIELD DSP pipeline.
    """

    def __init__(self, config: SCPIConfig):
        self.config = config
        hw = config.hardware_type.lower().replace(" ", "_").replace("-", "_")
        self.commands = ACQUIRE_COMMANDS.get(hw, ACQUIRE_COMMANDS["generic"])
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._instrument_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> Optional["SCPIEdgeDriver"]:
        """Create driver from environment variables. Returns None if no SCPI_HOST set."""
        host = os.environ.get("SCPI_HOST")
        if not host:
            return None

        config = SCPIConfig(
            host=host,
            port=int(os.environ.get("SCPI_PORT", "5000")),
            hardware_type=os.environ.get("SCPI_HARDWARE_TYPE", "generic"),
            node_id=os.environ.get("CEFIELD_NODE_ID", f"scpi-{host}"),
            lab_name=os.environ.get("CEFIELD_LAB_NAME"),
            lat=float(lat) if (lat := os.environ.get("CEFIELD_LAT")) else None,
            lon=float(lon) if (lon := os.environ.get("CEFIELD_LON")) else None,
            f0_estimate_hz=float(os.environ.get("CEFIELD_F0_HZ", "1.5e9")),
        )
        return cls(config)

    @property
    def is_connected(self) -> bool:
        return self._connected and self._sock is not None

    def connect(self) -> bool:
        """Connect to the SCPI instrument over TCP."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self.config.timeout)
            self._sock.connect((self.config.host, self.config.port))
            self._connected = True

            # Identify the instrument
            self._instrument_id = self._query(self.commands["idn"])
            logger.info(
                f"[SCPI-EDGE] Connected: {self._instrument_id} "
                f"@ {self.config.host}:{self.config.port}"
            )
            return True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.error(f"[SCPI-EDGE] Connection failed: {e}")
            self._cleanup()
            return False

    def disconnect(self) -> None:
        """Disconnect from instrument."""
        self._cleanup()
        self._connected = False
        self._instrument_id = None

    def _cleanup(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _send(self, cmd: str) -> None:
        if not self.is_connected:
            raise ConnectionError("Not connected to SCPI instrument")
        self._sock.sendall(f"{cmd}\r\n".encode())

    def _query(self, cmd: str) -> str:
        self._send(cmd)
        return self._sock.recv(self.config.buffer_size).decode().strip()

    def acquire(self, timeout: float = 10.0) -> np.ndarray:
        """
        Acquire a single waveform from the connected instrument.

        Returns:
            1D numpy float64 array of voltage samples
        """
        # Configure
        self._send(self.commands["format"])
        self._send(self.commands["decimate"])
        self._send(self.commands["trigger"])

        # Start acquisition
        self._send(self.commands["start"])

        # Poll for completion
        t0 = time.monotonic()
        while (time.monotonic() - t0) < timeout:
            status = self._query(self.commands["status"])
            if status.upper() in ("TD", "1", "DONE", "COMPLETE"):
                break
            time.sleep(0.05)
        else:
            self._send(self.commands["stop"])
            raise TimeoutError(f"Acquisition timeout ({timeout}s), status: {status}")

        # Read data
        raw = self._query(self.commands["data"])
        self._send(self.commands["stop"])

        # Parse
        cleaned = raw.strip().strip("{}")
        values = [float(x) for x in cleaned.split(",") if x.strip()]
        logger.info(f"[SCPI-EDGE] Acquired {len(values)} samples")
        return np.array(values, dtype=np.float64)

    def extract_signature(self, signal: np.ndarray) -> list[float]:
        """
        Run the CEFIELD Hilbert DSP pipeline on a raw waveform.
        Returns a 128-dimensional signature vector.
        """
        analytic = hilbert(signal)
        envelope = np.abs(analytic)
        indices = np.linspace(0, len(envelope) - 1, 128, dtype=int)
        return envelope[indices].tolist()

    def estimate_q_factor(self, signal: np.ndarray) -> float:
        """
        Estimate Q-factor from a time-domain waveform using the
        envelope decay method:
            Q ≈ π · f₀ · τ
        where τ is the 1/e decay time of the amplitude envelope.
        """
        envelope = np.abs(hilbert(signal))
        peak_idx = np.argmax(envelope)

        if peak_idx >= len(envelope) - 10:
            return 0.0

        decay_region = envelope[peak_idx:]
        peak_val = decay_region[0]

        if peak_val < 1e-12:
            return 0.0

        # Find 1/e point
        target = peak_val / np.e
        below_target = np.where(decay_region <= target)[0]

        if len(below_target) == 0:
            return 0.0

        e_idx = below_target[0]
        # Estimate sample rate from signal length (assuming 1ms window)
        sample_rate = len(signal) / 1e-3
        tau = e_idx / sample_rate

        q_factor = np.pi * self.config.f0_estimate_hz * tau
        return float(q_factor)

    def acquire_and_process(self) -> dict:
        """
        Full acquisition → processing pipeline.
        Returns a dict ready for CEFIELD API /api/v1/ingest.
        """
        signal = self.acquire()
        vector = self.extract_signature(signal)
        q_factor = self.estimate_q_factor(signal)

        return {
            "node_id": self.config.node_id,
            "hardware_type": self.config.hardware_type,
            "f0": self.config.f0_estimate_hz,
            "q_factor": q_factor,
            "signature_vector": vector,
            "lab_name": self.config.lab_name,
            "lat": self.config.lat,
            "lon": self.config.lon,
        }

    def get_status(self) -> dict:
        """Health status for monitoring."""
        return {
            "connected": self.is_connected,
            "host": self.config.host,
            "port": self.config.port,
            "hardware_type": self.config.hardware_type,
            "instrument_id": self._instrument_id,
            "node_id": self.config.node_id,
        }
