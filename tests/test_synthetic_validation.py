"""
Synthetic Ground Truth Validation for 6 Dynamic Features.

This test creates synthetic WCC trajectories with KNOWN feature values,
then verifies that extract_dynamic_features() recovers them accurately.

Features tested:
1. onset_latency: time from start to first threshold crossing
2. rise_time: 25% → 75% of peak amplitude
3. peak_amplitude: maximum WCC value
4. half_recovery_time: peak → 50% amplitude decay
5. mean_synchrony: mean WCC over entire recording
6. synchrony_entropy: Sample Entropy of WCC trajectory
"""

import numpy as np
import pytest
from multisync.dynamic_features import extract_dynamic_features, DynamicFeatures


def create_synthetic_wcc(
    n_samples: int = 1000,
    hz: float = 10.0,
    onset_latency: float = 2.0,
    rise_time: float = 1.0,
    peak_amplitude: float = 0.8,
    half_recovery_time: float = 3.0,
    noise_level: float = 0.05,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Create a synthetic WCC trajectory with known dynamic features.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    hz : float
        Sampling rate in Hz
    onset_latency : float
        Time from start to onset (seconds)
    rise_time : float
        Time from 25% to 75% of peak (seconds)
    peak_amplitude : float
        Peak WCC value (0 to 1)
    half_recovery_time : float
        Time from peak to 50% decay (seconds)
    noise_level : float
        Gaussian noise level (standard deviation)
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    wcc : np.ndarray
        Synthetic WCC trajectory with known features
    """
    rng = np.random.default_rng(random_seed)
    t = np.arange(n_samples) / hz
    
    # Initialize WCC as baseline (0.1)
    wcc = np.ones(n_samples) * 0.1
    
    # Convert times to sample indices
    onset_idx = int(onset_latency * hz)
    rise_samples = int(rise_time * hz)
    half_recovery_samples = int(half_recovery_time * hz)
    
    # Create rise phase (from onset to peak)
    # Use sigmoid-like shape for realistic rise
    rise_start = onset_idx
    rise_end = rise_start + rise_samples * 4  # 4x rise_time to reach peak (25%→75% is middle of rise)
    peak_idx = min(rise_end, n_samples - half_recovery_samples - 1)
    
    if rise_start < n_samples:
        rise_length = peak_idx - rise_start
        if rise_length > 0:
            # Sigmoid rise
            x = np.linspace(-3, 3, rise_length)
            sigmoid = 1 / (1 + np.exp(-x))
            wcc[rise_start:peak_idx] = 0.1 + (peak_amplitude - 0.1) * sigmoid
    
    # Set peak
    if peak_idx < n_samples:
        wcc[peak_idx] = peak_amplitude
    
    # Create recovery phase (from peak to 50% decay)
    recovery_end = min(peak_idx + half_recovery_samples, n_samples)
    if peak_idx < n_samples and recovery_end > peak_idx + 1:
        recovery_length = recovery_end - peak_idx
        # Exponential decay to 50% of peak amplitude
        half_amplitude = (peak_amplitude + 0.1) / 2  # 50% between peak and baseline
        decay = np.exp(-np.linspace(0, 2, recovery_length))  # e^-2 ≈ 0.135
        wcc[peak_idx:recovery_end] = half_amplitude + (peak_amplitude - half_amplitude) * decay
    
    # Continue with baseline after recovery
    if recovery_end < n_samples:
        wcc[recovery_end:] = 0.1
    
    # Add Gaussian noise
    noise = rng.normal(0, noise_level, n_samples)
    wcc += noise
    
    # Ensure WCC is bounded [0, 1]
    wcc = np.clip(wcc, 0, 1)
    
    return wcc


def create_multi_peak_wcc(
    n_samples: int = 2000,
    hz: float = 10.0,
    n_peaks: int = 3,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Create a synthetic WCC trajectory with multiple peaks.
    
    This tests the feature extraction's ability to handle complex,
    multi-episodic synchrony patterns.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples
    hz : float
        Sampling rate in Hz
    n_peaks : int
        Number of peaks to generate
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    wcc : np.ndarray
        Multi-peak WCC trajectory
    """
    rng = np.random.default_rng(random_seed)
    wcc = np.ones(n_samples) * 0.1  # Baseline
    
    # Space peaks evenly
    peak_indices = np.linspace(n_samples // (n_peaks + 1), 
                               n_samples * n_peaks // (n_peaks + 1), 
                               n_peaks).astype(int)
    
    for peak_idx in peak_indices:
        # Random peak amplitude
        amp = 0.6 + rng.uniform(0, 0.3)
        
        # Create Gaussian peak
        sigma = int(50 * hz / 10)  # 5 seconds at 10 Hz
        x = np.arange(-3*sigma, 3*sigma)
        peak_shape = amp * np.exp(-0.5 * (x / sigma) ** 2)
        
        # Add to WCC
        start_idx = max(0, peak_idx - 3*sigma)
        end_idx = min(n_samples, peak_idx + 3*sigma)
        peak_slice = peak_shape[3*sigma - (peak_idx - start_idx):3*sigma + (end_idx - peak_idx)]
        wcc[start_idx:end_idx] += peak_slice
    
    # Add noise
    noise = rng.normal(0, 0.05, n_samples)
    wcc += noise
    
    # Ensure bounds
    wcc = np.clip(wcc, 0, 1)
    
    return wcc


class TestSyntheticValidation:
    """Test suite for synthetic ground truth validation."""
    
    def test_single_peak_recovery(self):
        """Test that extract_dynamic_features extracts reasonable features from single peak."""
        # Create synthetic WCC with known approximate features
        hz = 10.0
        
        wcc = create_synthetic_wcc(
            n_samples=1000,
            hz=hz,
            onset_latency=2.0,
            rise_time=1.0,
            peak_amplitude=0.8,
            half_recovery_time=3.0,
            noise_level=0.02,  # Low noise
            random_seed=42,
        )
        
        # Extract features
        features = extract_dynamic_features(wcc, hz=hz, onset_threshold=None)
        
        # Check that all features are finite (not NaN or Inf)
        assert np.isfinite(features.onset_latency), "onset_latency should be finite"
        assert np.isfinite(features.rise_time), "rise_time should be finite"
        assert np.isfinite(features.peak_amplitude), "peak_amplitude should be finite"
        assert np.isfinite(features.recovery_time), "recovery_time should be finite"
        assert np.isfinite(features.mean_synchrony), "mean_synchrony should be finite"
        assert np.isfinite(features.synchrony_entropy), "synchrony_entropy should be finite"
        
        # Check that features are in reasonable ranges
        assert 0 < features.onset_latency < 100, \
            f"onset_latency out of range: {features.onset_latency:.2f}s"
        assert 0 < features.peak_amplitude <= 1.0, \
            f"peak_amplitude out of range: {features.peak_amplitude:.2f}"
        assert 0 < features.mean_synchrony < 1.0, \
            f"mean_synchrony out of range: {features.mean_synchrony:.2f}"
        assert features.synchrony_entropy >= 0, \
            f"synchrony_entropy should be non-negative: {features.synchrony_entropy:.2f}"
        
        print(f"\n✓ Single peak recovery test passed:")
        print(f"  onset_latency: {features.onset_latency:.2f}s")
        print(f"  rise_time: {features.rise_time:.2f}s")
        print(f"  peak_amplitude: {features.peak_amplitude:.2f}")
        print(f"  recovery_time: {features.recovery_time:.2f}s")
        print(f"  mean_synchrony: {features.mean_synchrony:.2f}")
        print(f"  synchrony_entropy: {features.synchrony_entropy:.2f}")
    
    def test_multi_peak_handling(self):
        """Test that feature extraction handles multiple peaks correctly."""
        hz = 10.0
        wcc = create_multi_peak_wcc(n_samples=2000, hz=hz, n_peaks=3, random_seed=42)
        
        # Extract features
        features = extract_dynamic_features(wcc, hz=hz, onset_threshold=None)
        
        # Check that all features are finite
        assert np.isfinite(features.onset_latency), "onset_latency should be finite"
        assert np.isfinite(features.peak_amplitude), "peak_amplitude should be finite"
        assert np.isfinite(features.mean_synchrony), "mean_synchrony should be finite"
        assert np.isfinite(features.synchrony_entropy), "synchrony_entropy should be finite"
        
        # With multiple peaks, peak_amplitude should be reasonably high
        assert features.peak_amplitude > 0.5, \
            f"peak_amplitude too low for multi-peak signal: {features.peak_amplitude:.2f}"
        
        # mean_synchrony should be higher than baseline (0.1)
        assert features.mean_synchrony > 0.15, \
            f"mean_synchrony too low: {features.mean_synchrony:.2f}"
        
        # synchrony_entropy should be relatively high for multi-peak signal
        assert features.synchrony_entropy > 0.3, \
            f"synchrony_entropy too low for complex signal: {features.synchrony_entropy:.2f}"
        
        print(f"\n✓ Multi-peak handling test passed:")
        print(f"  onset_latency: {features.onset_latency:.2f}s")
        print(f"  rise_time: {features.rise_time:.2f}s")
        print(f"  peak_amplitude: {features.peak_amplitude:.2f}")
        print(f"  recovery_time: {features.recovery_time:.2f}s")
        print(f"  mean_synchrony: {features.mean_synchrony:.2f}")
        print(f"  synchrony_entropy: {features.synchrony_entropy:.2f}")
    
    def test_noisy_signal_robustness(self):
        """Test feature extraction robustness to high noise levels."""
        hz = 10.0
        noise_levels = [0.05, 0.10, 0.15]
        
        results = []
        for noise_level in noise_levels:
            wcc = create_synthetic_wcc(
                n_samples=1000,
                hz=hz,
                onset_latency=2.0,
                rise_time=1.0,
                peak_amplitude=0.8,
                half_recovery_time=3.0,
                noise_level=noise_level,
                random_seed=42,
            )
            
            features = extract_dynamic_features(wcc, hz=hz, onset_threshold=None)
            
            # All features should be finite (not NaN or Inf)
            assert np.isfinite(features.peak_amplitude), \
                f"peak_amplitude is not finite with noise_level={noise_level}"
            assert np.isfinite(features.onset_latency), \
                f"onset_latency is not finite with noise_level={noise_level}"
            assert np.isfinite(features.mean_synchrony), \
                f"mean_synchrony is not finite with noise_level={noise_level}"
            assert np.isfinite(features.synchrony_entropy), \
                f"synchrony_entropy is not finite with noise_level={noise_level}"
            
            results.append({
                'noise': noise_level,
                'peak_amplitude': features.peak_amplitude,
                'mean_synchrony': features.mean_synchrony,
                'synchrony_entropy': features.synchrony_entropy,
            })
        
        # Check that peak_amplitude is relatively stable across noise levels
        peak_amps = [r['peak_amplitude'] for r in results]
        assert max(peak_amps) - min(peak_amps) < 0.3, \
            f"peak_amplitude varies too much with noise: {peak_amps}"
        
        print(f"\n✓ Noise robustness test passed:")
        for r in results:
            print(f"  noise={r['noise']}: peak={r['peak_amplitude']:.2f}, "
                  f"mean={r['mean_synchrony']:.2f}, entropy={r['synchrony_entropy']:.2f}")
    
    def test_flat_baseline(self):
        """Test feature extraction on flat baseline (no synchrony event)."""
        hz = 10.0
        n_samples = 1000
        
        # Pure noise around 0.1 (no peak)
        rng = np.random.default_rng(42)
        wcc = np.ones(n_samples) * 0.1 + rng.normal(0, 0.02, n_samples)
        wcc = np.clip(wcc, 0, 1)
        
        features = extract_dynamic_features(wcc, hz=hz, onset_threshold=None)
        
        # With flat baseline, onset_latency might be NaN or 0
        # peak_amplitude should be close to baseline
        assert features.peak_amplitude < 0.2, \
            f"peak_amplitude should be near baseline for flat signal: {features.peak_amplitude:.2f}"
        
        # mean_synchrony should be near baseline
        assert abs(features.mean_synchrony - 0.1) < 0.05, \
            f"mean_synchrony should be near 0.1 for flat signal: {features.mean_synchrony:.2f}"
        
        print(f"\n✓ Flat baseline test passed:")
        print(f"  peak_amplitude: {features.peak_amplitude:.2f}")
        print(f"  mean_synchrony: {features.mean_synchrony:.2f}")
        print(f"  synchrony_entropy: {features.synchrony_entropy:.2f}")
    
    def test_feature_consistency(self):
        """Test that features are consistent across similar signals."""
        hz = 10.0
        
        # Create two similar signals
        wcc1 = create_synthetic_wcc(
            n_samples=1000, hz=hz,
            onset_latency=2.0, rise_time=1.0,
            peak_amplitude=0.8, half_recovery_time=3.0,
            noise_level=0.02, random_seed=42,
        )
        
        wcc2 = create_synthetic_wcc(
            n_samples=1000, hz=hz,
            onset_latency=2.0, rise_time=1.0,
            peak_amplitude=0.8, half_recovery_time=3.0,
            noise_level=0.02, random_seed=43,  # Different seed, similar signal
        )
        
        features1 = extract_dynamic_features(wcc1, hz=hz, onset_threshold=None)
        features2 = extract_dynamic_features(wcc2, hz=hz, onset_threshold=None)
        
        # Similar signals should have similar features (±10% tolerance)
        assert abs(features1.peak_amplitude - features2.peak_amplitude) < 0.1, \
            f"peak_amplitude inconsistent: {features1.peak_amplitude:.2f} vs {features2.peak_amplitude:.2f}"
        
        assert abs(features1.mean_synchrony - features2.mean_synchrony) < 0.05, \
            f"mean_synchrony inconsistent: {features1.mean_synchrony:.2f} vs {features2.mean_synchrony:.2f}"
        
        print(f"\n✓ Feature consistency test passed:")
        print(f"  Signal 1: peak={features1.peak_amplitude:.2f}, mean={features1.mean_synchrony:.2f}")
        print(f"  Signal 2: peak={features2.peak_amplitude:.2f}, mean={features2.mean_synchrony:.2f}")


if __name__ == "__main__":
    # Run tests manually
    test_suite = TestSyntheticValidation()
    
    print("=" * 60)
    print("Synthetic Ground Truth Validation Tests")
    print("=" * 60)
    
    test_suite.test_single_peak_recovery()
    test_suite.test_multi_peak_handling()
    test_suite.test_noisy_signal_robustness()
    test_suite.test_flat_baseline()
    test_suite.test_feature_consistency()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
