"""
CEFIELD SCPI Hardware Bridge
=============================
Production-grade SCPI (Standard Commands for Programmable Instruments)
bridge that enables direct instrument → CEFIELD pipeline integration.

Supported hardware families:
  • Red Pitaya STEMlab 125-14 / 250-12  (native SCPI over TCP)
  • Keysight / Agilent DSO/MSO series   (SCPI over TCP/VISA)
  • PicoScope 4000/5000/6000 series     (SCPI over TCP)
  • Rohde & Schwarz RTx series          (SCPI over TCP/VISA)
  • Generic SCPI-compliant instruments   (configurable command set)

Architecture:
  Physical Instrument ←→ TCP Socket ←→ SCPIBridge ←→ DSP Pipeline ←→ CEFIELD API

The bridge handles:
  1. Connection management (connect / disconnect / reconnect)
  2. Command abstraction (device-specific → unified interface)
  3. Raw data acquisition (time-domain waveform capture)
  4. Health monitoring (connection state + instrument status)
"""
from __future__ import annotations

import socket
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger("cefield.scpi")


# ─── Hardware Command Profiles ───────────────────────────────────────────────────
class SCPIHardwareType(str, Enum):
    RED_PITAYA = "red_pitaya"
    KEYSIGHT = "keysight"
    PICOSCOPE = "picoscope"
    ROHDE_SCHWARZ = "rohde_schwarz"
    GENERIC = "generic"


@dataclass
class SCPICommandSet:
    """Device-specific SCPI command mappings."""
    idn: str = "*IDN?"                           # Identification query
    reset: str = "*RST"                           # Reset to default state
    acquire_start: str = "ACQ:START"              # Start acquisition
    acquire_stop: str = "ACQ:STOP"                # Stop acquisition
    acquire_status: str = "ACQ:STATUS?"           # Query acquisition status
    data_format: str = "ACQ:DATA:FORMAT ASCII"    # Set data format
    data_query: str = "ACQ:SOUR1:DATA?"           # Query waveform data
    sample_rate: str = "ACQ:DEC 1"                # Set decimation / sample rate
    trigger_mode: str = "ACQ:TRIG:LEV 0"          # Trigger level
    trigger_source: str = "ACQ:TRIG CH1"          # Trigger source


# Pre-configured command sets per hardware family
HARDWARE_COMMANDS: dict[SCPIHardwareType, SCPICommandSet] = {
    SCPIHardwareType.RED_PITAYA: SCPICommandSet(
        acquire_start="ACQ:START",
        acquire_stop="ACQ:STOP",
        acquire_status="ACQ:TRIG:STAT?",
        data_format="ACQ:DATA:FORMAT ASCII",
        data_query="ACQ:SOUR1:DATA?",
        sample_rate="ACQ:DEC 1",
        trigger_mode="ACQ:TRIG:LEV 0",
        trigger_source="ACQ:TRIG NOW",
    ),
    SCPIHardwareType.KEYSIGHT: SCPICommandSet(
        acquire_start=":DIGITIZE CHAN1",
        acquire_stop=":STOP",
        acquire_status=":OPER:COND?",
        data_format=":WAVEFORM:FORMAT ASCII",
        data_query=":WAVEFORM:DATA?",
        sample_rate=":TIMEBASE:RANGE 1E-3",
        trigger_mode=":TRIGGER:EDGE:LEVEL 0",
        trigger_source=":TRIGGER:EDGE:SOURCE CHAN1",
    ),
    SCPIHardwareType.PICOSCOPE: SCPICommandSet(
        acquire_start=":RUN",
        acquire_stop=":STOP",
        acquire_status=":STATUS?",
        data_format=":FORMAT:DATA ASCii",
        data_query=":FETCH:WAV?",
        sample_rate=":TIMEBASE:RANGE 1E-3",
        trigger_mode=":TRIGGER:MAIN:LEVEL 0",
        trigger_source=":TRIGGER:MAIN:SOURCE CH1",
    ),
    SCPIHardwareType.ROHDE_SCHWARZ: SCPICommandSet(
        acquire_start="RUN",
        acquire_stop="STOP",
        acquire_status="ACQ:STAT?",
        data_format="FORM ASC",
        data_query="CHAN1:DATA?",
        sample_rate="TIM:RANG 1E-3",
        trigger_mode="TRIG:LEV1 0",
        trigger_source="TRIG:SOUR CH1",
    ),
    SCPIHardwareType.GENERIC: SCPICommandSet(),
}


@dataclass
class SCPIConnectionConfig:
    """Configuration for an SCPI instrument connection."""
    host: str
    port: int = 5000
    hardware_type: SCPIHardwareType = SCPIHardwareType.GENERIC
    timeout_seconds: float = 5.0
    recv_buffer_size: int = 65536
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    node_id: Optional[str] = None       # CEFIELD node identifier
    lab_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


# ─── SCPI Bridge Core ────────────────────────────────────────────────────────────
class SCPIBridge:
    """
    Production SCPI bridge for direct instrument integration.

    Usage:
        config = SCPIConnectionConfig(
            host="192.168.1.100",
            port=5000,
            hardware_type=SCPIHardwareType.RED_PITAYA,
            node_id="lab-munich-rp-01",
        )
        bridge = SCPIBridge(config)
        bridge.connect()
        idn = bridge.identify()
        signal = bridge.acquire_waveform()
        bridge.disconnect()
    """

    def __init__(self, config: SCPIConnectionConfig):
        self.config = config
        self.commands = HARDWARE_COMMANDS.get(
            config.hardware_type, HARDWARE_COMMANDS[SCPIHardwareType.GENERIC]
        )
        self._socket: Optional[socket.socket] = None
        self._connected: bool = False
        self._instrument_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._socket is not None

    @property
    def instrument_id(self) -> Optional[str]:
        return self._instrument_id

    def connect(self) -> bool:
        """Establish TCP connection to SCPI instrument."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.config.timeout_seconds)
                self._socket.connect((self.config.host, self.config.port))
                self._connected = True
                logger.info(
                    f"[SCPI] Connected to {self.config.host}:{self.config.port} "
                    f"({self.config.hardware_type.value}) on attempt {attempt}"
                )
                return True
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                logger.warning(
                    f"[SCPI] Connection attempt {attempt}/{self.config.max_retries} "
                    f"to {self.config.host}:{self.config.port} failed: {e}"
                )
                self._cleanup_socket()
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_seconds)

        logger.error(
            f"[SCPI] Failed to connect to {self.config.host}:{self.config.port} "
            f"after {self.config.max_retries} attempts"
        )
        return False

    def disconnect(self) -> None:
        """Gracefully close the SCPI connection."""
        self._cleanup_socket()
        self._connected = False
        self._instrument_id = None
        logger.info(f"[SCPI] Disconnected from {self.config.host}")

    def _cleanup_socket(self) -> None:
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    def _send_command(self, command: str) -> None:
        """Send a SCPI command (no response expected)."""
        if not self.is_connected:
            raise ConnectionError("SCPI bridge not connected")
        self._socket.sendall(f"{command}\r\n".encode())

    def _query(self, command: str) -> str:
        """Send a SCPI query and return the response string."""
        if not self.is_connected:
            raise ConnectionError("SCPI bridge not connected")
        self._socket.sendall(f"{command}\r\n".encode())
        response = self._socket.recv(self.config.recv_buffer_size).decode().strip()
        return response

    def identify(self) -> str:
        """Query instrument identification (*IDN?)."""
        idn = self._query(self.commands.idn)
        self._instrument_id = idn
        logger.info(f"[SCPI] Instrument ID: {idn}")
        return idn

    def reset(self) -> None:
        """Reset instrument to default state (*RST)."""
        self._send_command(self.commands.reset)
        logger.info("[SCPI] Instrument reset to defaults")

    def configure_acquisition(
        self,
        sample_rate_cmd: Optional[str] = None,
        trigger_level_cmd: Optional[str] = None,
        trigger_source_cmd: Optional[str] = None,
        data_format_cmd: Optional[str] = None,
    ) -> None:
        """
        Configure acquisition parameters. Uses hardware defaults if not specified.
        """
        self._send_command(data_format_cmd or self.commands.data_format)
        self._send_command(sample_rate_cmd or self.commands.sample_rate)
        self._send_command(trigger_level_cmd or self.commands.trigger_mode)
        self._send_command(trigger_source_cmd or self.commands.trigger_source)
        logger.info("[SCPI] Acquisition configured")

    def acquire_waveform(self, timeout_seconds: float = 10.0) -> np.ndarray:
        """
        Acquire a single waveform from the instrument.

        Returns:
            1D numpy array of float64 voltage samples

        Raises:
            TimeoutError: If acquisition does not complete in time
            ConnectionError: If bridge is not connected
            ValueError: If response cannot be parsed
        """
        self._send_command(self.commands.acquire_start)

        # Wait for trigger / acquisition complete
        start = time.monotonic()
        while (time.monotonic() - start) < timeout_seconds:
            status = self._query(self.commands.acquire_status)
            if status.upper() in ("TD", "1", "DONE", "COMPLETE"):
                break
            time.sleep(0.05)
        else:
            self._send_command(self.commands.acquire_stop)
            raise TimeoutError(
                f"SCPI acquisition timeout after {timeout_seconds}s "
                f"(last status: {status})"
            )

        # Retrieve waveform data
        raw_response = self._query(self.commands.data_query)
        samples = self._parse_ascii_waveform(raw_response)

        self._send_command(self.commands.acquire_stop)

        logger.info(f"[SCPI] Acquired {len(samples)} samples")
        return samples

    @staticmethod
    def _parse_ascii_waveform(raw: str) -> np.ndarray:
        """
        Parse ASCII waveform response into numpy array.
        Handles common SCPI response formats:
          - Comma-separated: "0.123,0.456,0.789"
          - Curly-brace wrapped: "{0.123,0.456,0.789}"
          - Header prefixed: "#800001000<binary>" (detected and rejected)
        """
        cleaned = raw.strip().strip("{}")

        if cleaned.startswith("#"):
            raise ValueError(
                "Binary waveform format detected — set data format to ASCII "
                "before acquisition"
            )

        try:
            values = [float(x) for x in cleaned.split(",") if x.strip()]
        except ValueError as e:
            raise ValueError(f"Failed to parse SCPI waveform: {e}\nRaw: {raw[:200]}")

        if len(values) < 10:
            raise ValueError(
                f"Waveform too short ({len(values)} samples) — "
                f"check instrument configuration"
            )

        return np.array(values, dtype=np.float64)

    def get_health_status(self) -> dict:
        """Returns current bridge health status for monitoring."""
        return {
            "host": self.config.host,
            "port": self.config.port,
            "hardware_type": self.config.hardware_type.value,
            "connected": self.is_connected,
            "instrument_id": self._instrument_id,
            "node_id": self.config.node_id,
            "lab_name": self.config.lab_name,
        }


# ─── Factory ─────────────────────────────────────────────────────────────────────
def create_bridge_from_env() -> Optional[SCPIBridge]:
    """
    Create an SCPI bridge from environment variables.
    Expected env vars:
        SCPI_HOST          - Instrument IP/hostname
        SCPI_PORT          - TCP port (default: 5000)
        SCPI_HARDWARE_TYPE - One of: red_pitaya, keysight, picoscope, rohde_schwarz, generic
        CEFIELD_NODE_ID    - Node identifier for CEFIELD registration
        CEFIELD_LAB_NAME   - Human-readable lab name
        CEFIELD_LAT        - GPS latitude
        CEFIELD_LON        - GPS longitude
    """
    import os

    host = os.environ.get("SCPI_HOST")
    if not host:
        logger.info("[SCPI] No SCPI_HOST set — SCPI bridge disabled")
        return None

    hw_str = os.environ.get("SCPI_HARDWARE_TYPE", "generic").lower()
    try:
        hw_type = SCPIHardwareType(hw_str)
    except ValueError:
        logger.warning(f"[SCPI] Unknown hardware type '{hw_str}', falling back to generic")
        hw_type = SCPIHardwareType.GENERIC

    config = SCPIConnectionConfig(
        host=host,
        port=int(os.environ.get("SCPI_PORT", "5000")),
        hardware_type=hw_type,
        node_id=os.environ.get("CEFIELD_NODE_ID", f"scpi-{host}"),
        lab_name=os.environ.get("CEFIELD_LAB_NAME"),
        lat=float(os.environ.get("CEFIELD_LAT", "0")) or None,
        lon=float(os.environ.get("CEFIELD_LON", "0")) or None,
    )
    return SCPIBridge(config)
