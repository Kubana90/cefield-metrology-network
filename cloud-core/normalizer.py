"""
CEFIELD Hardware Normalization Layer
=====================================
Enables cross-hardware vector comparison in pgvector by mapping
device-specific signature vectors to a common reference basis.

Without this layer, an L2 distance between a Red Pitaya vector
and a Keysight DSO vector is physically meaningless.
With this layer, a precursor pattern from Stanford (Keysight)
can be matched against Munich (Red Pitaya) at scale.

Normalization pipeline per vector:
  1. SNR correction      (compensate for noise floor differences)
  2. Bandwidth weighting (scale spectral components by BW ratio)
  3. ADC resolution norm (effective bits correction)
  4. L2 unit-vector norm (hardware-agnostic cosine basis)
"""

from __future__ import annotations

import numpy as np

# Hardware calibration profiles — extend as new hardware types are onboarded
HARDWARE_PROFILES: dict[str, dict] = {
    "red_pitaya": {"adc_bits": 14, "bandwidth_hz": 50e6, "noise_floor_db": -120},
    "red_pitaya_stem": {"adc_bits": 14, "bandwidth_hz": 50e6, "noise_floor_db": -120},
    "pico_scope_6000": {"adc_bits": 12, "bandwidth_hz": 200e6, "noise_floor_db": -110},
    "pico_scope_4000": {"adc_bits": 12, "bandwidth_hz": 80e6, "noise_floor_db": -112},
    "keysight_dso": {"adc_bits": 16, "bandwidth_hz": 6e9, "noise_floor_db": -130},
    "keysight_mso": {"adc_bits": 16, "bandwidth_hz": 6e9, "noise_floor_db": -130},
    "rtl_sdr": {"adc_bits": 8, "bandwidth_hz": 3e6, "noise_floor_db": -100},
    "hackrf_one": {"adc_bits": 8, "bandwidth_hz": 20e6, "noise_floor_db": -103},
    "usrp_b200": {"adc_bits": 12, "bandwidth_hz": 56e6, "noise_floor_db": -115},
    "generic_sdr": {"adc_bits": 8, "bandwidth_hz": 3e6, "noise_floor_db": -100},
}

# Common reference class — all vectors are normalized to this basis
REFERENCE_CLASS: str = "generic_sdr"


def _get_profile(hardware_type: str) -> dict:
    key = hardware_type.lower().replace(" ", "_").replace("-", "_")
    return HARDWARE_PROFILES.get(key, HARDWARE_PROFILES[REFERENCE_CLASS])


def normalize_vector(
    vector: list[float],
    hardware_type: str,
    target_class: str = REFERENCE_CLASS,
) -> list[float]:
    """
    Normalize a hardware-specific 128-dim signature vector to the
    common reference basis for cross-hardware pgvector comparison.

    Args:
        vector       : Raw 128-dim signature from edge-node DSP pipeline
        hardware_type: Source hardware identifier string
        target_class : Reference hardware class (default: generic_sdr)

    Returns:
        128-dim L2-normalized float list compatible with pgvector L2 search
    """
    source = _get_profile(hardware_type)
    target = HARDWARE_PROFILES.get(target_class, HARDWARE_PROFILES[REFERENCE_CLASS])

    vec = np.array(vector, dtype=np.float64)
    n = len(vec)

    # Step 1: SNR correction (dB difference → linear amplitude scale)
    snr_delta_db = source["noise_floor_db"] - target["noise_floor_db"]
    vec = vec * (10 ** (snr_delta_db / 20.0))

    # Step 2: Bandwidth normalization
    bw_ratio = np.clip(source["bandwidth_hz"] / target["bandwidth_hz"], 0.1, 100.0)
    freq_weights = np.linspace(1.0, float(bw_ratio), n)
    vec = vec / (freq_weights + 1e-12)

    # Step 3: ADC resolution correction
    adc_scale = (2 ** target["adc_bits"]) / (2 ** source["adc_bits"])
    vec = vec * adc_scale

    # Step 4: L2 unit-vector normalization
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 1e-12 else vec.tolist()


def get_hardware_info(hardware_type: str) -> dict:
    """Returns hardware profile metadata for a given hardware type string."""
    profile = _get_profile(hardware_type)
    return {
        "hardware_type": hardware_type,
        "matched_profile": profile,
        "reference_class": REFERENCE_CLASS,
        "normalization_applied": hardware_type.lower().replace(" ", "_") != REFERENCE_CLASS,
    }


def list_supported_hardware() -> list[str]:
    """Returns all hardware types with built-in calibration profiles."""
    return list(HARDWARE_PROFILES.keys())
