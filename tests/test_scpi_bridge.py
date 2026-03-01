"""
CEFIELD SCPI Bridge — Test Suite
==================================
Tests for SCPI hardware bridge and edge-side driver.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

import sys
import os

# Ensure edge-node is importable
_EDGE_NODE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "edge-node")
)
if _EDGE_NODE not in sys.path:
    sys.path.insert(0, _EDGE_NODE)

from scpi_bridge import (
    SCPIBridge,
    SCPIConnectionConfig,
    SCPIHardwareType,
    SCPICommandSet,
    HARDWARE_COMMANDS,
    create_bridge_from_env,
)
from scpi_driver import SCPIEdgeDriver, SCPIConfig


# ═══════════════════════════════════════════════════════════════════════════════════
# SCPI Bridge Tests (cloud-core)
# ═══════════════════════════════════════════════════════════════════════════════════

class TestSCPICommandSets:
    """Verify all hardware command sets are complete."""

    def test_all_hardware_types_have_commands(self):
        for hw_type in SCPIHardwareType:
            assert hw_type in HARDWARE_COMMANDS

    def test_red_pitaya_specific_commands(self):
        cmds = HARDWARE_COMMANDS[SCPIHardwareType.RED_PITAYA]
        assert "ACQ:START" in cmds.acquire_start
        assert "ACQ:SOUR1:DATA?" in cmds.data_query
        assert "ACQ:TRIG NOW" in cmds.trigger_source

    def test_keysight_specific_commands(self):
        cmds = HARDWARE_COMMANDS[SCPIHardwareType.KEYSIGHT]
        assert ":DIGITIZE" in cmds.acquire_start
        assert ":WAVEFORM:DATA?" in cmds.data_query

    def test_rohde_schwarz_commands(self):
        cmds = HARDWARE_COMMANDS[SCPIHardwareType.ROHDE_SCHWARZ]
        assert cmds.acquire_start == "RUN"
        assert "CHAN1:DATA?" in cmds.data_query

    def test_generic_fallback_exists(self):
        cmds = HARDWARE_COMMANDS[SCPIHardwareType.GENERIC]
        assert cmds.idn == "*IDN?"
        assert cmds.reset == "*RST"


class TestSCPIBridgeConfig:
    """Test bridge configuration."""

    def test_default_config(self):
        config = SCPIConnectionConfig(host="192.168.1.100")
        assert config.port == 5000
        assert config.timeout_seconds == 5.0
        assert config.hardware_type == SCPIHardwareType.GENERIC
        assert config.max_retries == 3

    def test_custom_config(self):
        config = SCPIConnectionConfig(
            host="10.0.0.1",
            port=9999,
            hardware_type=SCPIHardwareType.RED_PITAYA,
            timeout_seconds=10.0,
            node_id="lab-custom-01",
            lab_name="Custom Lab",
            lat=48.13,
            lon=11.58,
        )
        assert config.port == 9999
        assert config.hardware_type == SCPIHardwareType.RED_PITAYA
        assert config.node_id == "lab-custom-01"


class TestSCPIBridgeCore:
    """Test bridge core functionality."""

    def test_initial_state(self):
        config = SCPIConnectionConfig(host="localhost")
        bridge = SCPIBridge(config)
        assert bridge.is_connected is False
        assert bridge.instrument_id is None

    def test_health_status_disconnected(self):
        config = SCPIConnectionConfig(
            host="192.168.1.100",
            hardware_type=SCPIHardwareType.KEYSIGHT,
            node_id="test-node",
        )
        bridge = SCPIBridge(config)
        status = bridge.get_health_status()
        assert status["connected"] is False
        assert status["hardware_type"] == "keysight"
        assert status["node_id"] == "test-node"

    def test_disconnect_from_disconnected_state(self):
        config = SCPIConnectionConfig(host="localhost")
        bridge = SCPIBridge(config)
        bridge.disconnect()  # Should not raise
        assert bridge.is_connected is False

    def test_parse_ascii_waveform_comma_separated(self):
        raw = "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0," + ",".join(["0.5"] * 90)
        result = SCPIBridge._parse_ascii_waveform(raw)
        assert isinstance(result, np.ndarray)
        assert len(result) == 100
        assert result.dtype == np.float64

    def test_parse_ascii_waveform_curly_braces(self):
        values = ",".join([f"{i * 0.01:.3f}" for i in range(100)])
        raw = "{" + values + "}"
        result = SCPIBridge._parse_ascii_waveform(raw)
        assert len(result) == 100

    def test_parse_binary_format_raises(self):
        raw = "#800001000" + "x" * 1000
        with pytest.raises(ValueError, match="Binary waveform"):
            SCPIBridge._parse_ascii_waveform(raw)

    def test_parse_too_short_raises(self):
        raw = "0.1,0.2,0.3"
        with pytest.raises(ValueError, match="too short"):
            SCPIBridge._parse_ascii_waveform(raw)


class TestSCPIBridgeFactory:
    """Test environment-based bridge creation."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_host_returns_none(self):
        result = create_bridge_from_env()
        assert result is None

    @patch.dict(os.environ, {
        "SCPI_HOST": "192.168.1.100",
        "SCPI_PORT": "5555",
        "SCPI_HARDWARE_TYPE": "red_pitaya",
        "CEFIELD_NODE_ID": "test-rp-01",
        "CEFIELD_LAB_NAME": "Test Lab",
    })
    def test_creates_bridge_from_env(self):
        bridge = create_bridge_from_env()
        assert bridge is not None
        assert bridge.config.host == "192.168.1.100"
        assert bridge.config.port == 5555
        assert bridge.config.hardware_type == SCPIHardwareType.RED_PITAYA

    @patch.dict(os.environ, {
        "SCPI_HOST": "10.0.0.1",
        "SCPI_HARDWARE_TYPE": "unknown_device",
    })
    def test_unknown_hardware_falls_back_to_generic(self):
        bridge = create_bridge_from_env()
        assert bridge is not None
        assert bridge.config.hardware_type == SCPIHardwareType.GENERIC


# ═══════════════════════════════════════════════════════════════════════════════════
# SCPI Edge Driver Tests (edge-node)
# ═══════════════════════════════════════════════════════════════════════════════════

class TestSCPIEdgeDriver:
    """Test edge-side SCPI driver."""

    def test_initial_state(self):
        config = SCPIConfig(host="localhost")
        driver = SCPIEdgeDriver(config)
        assert driver.is_connected is False

    def test_extract_signature_produces_128_dims(self):
        config = SCPIConfig(host="localhost")
        driver = SCPIEdgeDriver(config)
        signal = np.sin(np.linspace(0, 100, 10000))
        vector = driver.extract_signature(signal)
        assert len(vector) == 128
        assert all(isinstance(v, float) for v in vector)

    def test_estimate_q_factor_decaying_signal(self):
        config = SCPIConfig(host="localhost", f0_estimate_hz=1.5e9)
        driver = SCPIEdgeDriver(config)
        t = np.linspace(0, 1e-3, 10000)
        signal = np.exp(-t / 5e-5) * np.sin(2 * np.pi * 1e6 * t)
        q = driver.estimate_q_factor(signal)
        assert q > 0

    def test_estimate_q_factor_flat_signal_returns_zero(self):
        config = SCPIConfig(host="localhost")
        driver = SCPIEdgeDriver(config)
        signal = np.zeros(10000)
        q = driver.estimate_q_factor(signal)
        assert q == 0.0

    def test_get_status(self):
        config = SCPIConfig(
            host="192.168.1.200",
            hardware_type="red_pitaya",
            node_id="edge-rp-01",
        )
        driver = SCPIEdgeDriver(config)
        status = driver.get_status()
        assert status["connected"] is False
        assert status["hardware_type"] == "red_pitaya"
        assert status["node_id"] == "edge-rp-01"

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_returns_none_without_host(self):
        result = SCPIEdgeDriver.from_env()
        assert result is None

    @patch.dict(os.environ, {
        "SCPI_HOST": "10.0.0.50",
        "SCPI_PORT": "4000",
        "SCPI_HARDWARE_TYPE": "keysight",
        "CEFIELD_NODE_ID": "edge-ks-01",
        "CEFIELD_LAB_NAME": "Keysight Lab",
        "CEFIELD_F0_HZ": "2.4e9",
    })
    def test_from_env_creates_driver(self):
        driver = SCPIEdgeDriver.from_env()
        assert driver is not None
        assert driver.config.host == "10.0.0.50"
        assert driver.config.port == 4000
        assert driver.config.hardware_type == "keysight"
        assert driver.config.f0_estimate_hz == 2.4e9

    def test_acquire_and_process_output_structure(self):
        """Verify acquire_and_process returns CEFIELD-compatible payload."""
        config = SCPIConfig(
            host="localhost",
            node_id="test-node",
            hardware_type="red_pitaya",
            lab_name="Test Lab",
            lat=48.13,
            lon=11.58,
            f0_estimate_hz=1.5e9,
        )
        driver = SCPIEdgeDriver(config)

        # Mock the acquire method to return synthetic data
        t = np.linspace(0, 1e-3, 10000)
        mock_signal = np.exp(-t / 5e-5) * np.sin(2 * np.pi * 1e6 * t)
        driver.acquire = MagicMock(return_value=mock_signal)

        result = driver.acquire_and_process()

        assert result["node_id"] == "test-node"
        assert result["hardware_type"] == "red_pitaya"
        assert result["f0"] == 1.5e9
        assert len(result["signature_vector"]) == 128
        assert result["q_factor"] > 0
        assert result["lab_name"] == "Test Lab"
        assert result["lat"] == 48.13
        assert result["lon"] == 11.58
