import numpy as np
from scipy.signal import hilbert, find_peaks

class SignalProcessor:
    """
    Handles local Edge-computation so raw data never has to leave the lab.
    """
    @staticmethod
    def extract_signature(time_array: np.ndarray, voltage_array: np.ndarray, target_dim: int = 128) -> list[float]:
        """
        Compresses the raw RF ring-down into a standardized latent vector.
        """
        # 1. Analytic signal via Hilbert Transform to get the envelope
        analytic_signal = hilbert(voltage_array)
        amplitude_envelope = np.abs(analytic_signal)
        
        # 2. Downsample/Interpolate to exact target dimensions
        indices = np.linspace(0, len(amplitude_envelope) - 1, target_dim, dtype=int)
        signature = amplitude_envelope[indices]
        
        # 3. Normalize vector (0 to 1) to make it hardware-agnostic for the Global DB
        max_val = np.max(signature)
        if max_val > 0:
            signature = signature / max_val
            
        return signature.tolist()

    @staticmethod
    def estimate_q_factor(time_array: np.ndarray, voltage_array: np.ndarray, f0: float) -> float:
        """
        Rough local estimation of Q-factor using exponential decay fit on the envelope.
        """
        analytic_signal = hilbert(voltage_array)
        envelope = np.abs(analytic_signal)
        
        # Simple log-linear fit for decay time (tau)
        # log(A) = -t/tau + C
        # We only fit the first 30% of the signal to avoid noise floor issues
        fit_length = int(len(time_array) * 0.3)
        valid_indices = envelope[:fit_length] > 0
        
        t_fit = time_array[:fit_length][valid_indices]
        log_env = np.log(envelope[:fit_length][valid_indices])
        
        if len(t_fit) < 2:
            return 0.0
            
        slope, _ = np.polyfit(t_fit, log_env, 1)
        
        if slope >= 0:
            return 0.0 # Failed fit or growing signal
            
        tau = -1.0 / slope
        q_factor = np.pi * f0 * tau
        return float(q_factor)
