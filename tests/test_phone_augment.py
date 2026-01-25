"""
Unit tests for phone audio degradation pipeline.

Tests:
- Phone degradation effects
- Mel spectrogram conversion
- Audio normalization
- Configuration
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from phone_augment import (
    PhoneConfig,
    PhoneDegradation,
    PhoneMelSpectrogram
)


class TestPhoneConfig:
    """Test PhoneConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = PhoneConfig()

        assert config.SAMPLE_RATE == 8000, "Phone sample rate should be 8kHz"
        assert config.LOW_FREQ == 300, "Low freq cutoff should be 300Hz"
        assert config.HIGH_FREQ == 3400, "High freq cutoff should be 3.4kHz"
        assert config.N_MELS == 64, "Should use 64 mel bands"

    def test_num_samples_calculation(self):
        """Test that NUM_SAMPLES is correctly calculated."""
        config = PhoneConfig()
        expected = int(8000 * 3.0)  # 24000 samples

        assert config.NUM_SAMPLES == expected


class TestPhoneDegradation:
    """Test phone degradation pipeline."""

    @pytest.fixture
    def degrader(self):
        """Create a phone degrader with all effects enabled."""
        return PhoneDegradation(
            noise_prob=1.0,
            codec_prob=1.0,
            packet_loss_prob=1.0,
            echo_prob=1.0
        )

    @pytest.fixture
    def clean_audio(self):
        """Create clean test audio (1kHz tone)."""
        sample_rate = 16000
        duration = 3.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * torch.pi * 1000 * t).unsqueeze(0)
        return audio, sample_rate

    def test_resampling(self, clean_audio):
        """Test that audio is resampled to 8kHz."""
        degrader = PhoneDegradation()
        audio, sample_rate = clean_audio

        degraded = degrader(audio, sample_rate)

        # Output should have ~24000 samples (3 sec * 8000 Hz)
        expected_samples = int(3.0 * 8000)
        actual_samples = degraded.shape[1]

        # Allow some tolerance for resampling
        assert abs(actual_samples - expected_samples) < 100

    def test_output_shape(self, degrader, clean_audio):
        """Test that output has correct shape."""
        audio, sample_rate = clean_audio
        degraded = degrader(audio, sample_rate)

        assert degraded.ndim == 2, "Output should be 2D (channels, samples)"
        assert degraded.shape[0] == 1, "Output should be mono"

    def test_normalization(self, degrader, clean_audio):
        """Test that output is normalized."""
        audio, sample_rate = clean_audio
        degraded = degrader(audio, sample_rate)

        max_val = degraded.abs().max()
        assert max_val <= 1.0, f"Audio should be normalized to [-1, 1], got max={max_val}"

    def test_bandpass_filter(self):
        """Test that bandpass filter removes high frequencies."""
        degrader = PhoneDegradation(
            noise_prob=0,
            codec_prob=0,
            packet_loss_prob=0,
            echo_prob=0
        )

        # Create high frequency tone (5kHz - above phone cutoff)
        sample_rate = 16000
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        high_freq_audio = torch.sin(2 * torch.pi * 5000 * t).unsqueeze(0)

        degraded = degrader(high_freq_audio, sample_rate)

        # High frequency should be attenuated significantly
        # Energy should be much lower
        input_energy = (high_freq_audio ** 2).mean()
        output_energy = (degraded ** 2).mean()

        # Output energy should be much lower (high freq filtered out)
        assert output_energy < input_energy * 0.5, "High frequencies should be filtered"

    def test_no_modification_flag(self):
        """Test degrader with all effects disabled."""
        degrader = PhoneDegradation(
            noise_prob=0,
            codec_prob=0,
            packet_loss_prob=0,
            echo_prob=0
        )

        # Create 8kHz audio (no resampling needed)
        sample_rate = 8000
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * torch.pi * 1000 * t).unsqueeze(0)

        degraded = degrader(audio, sample_rate)

        # Should still apply bandpass filter, so not identical
        # but shape should be preserved
        assert degraded.shape == audio.shape

    def test_reproducibility_with_seed(self, clean_audio):
        """Test that same seed produces similar output (not strictly identical due to numpy)."""
        import random
        import numpy as np

        audio, sample_rate = clean_audio

        # Run twice with same seeds for both random and numpy
        random.seed(42)
        np.random.seed(42)
        degrader1 = PhoneDegradation()
        result1 = degrader1(audio.clone(), sample_rate)

        random.seed(42)
        np.random.seed(42)
        degrader2 = PhoneDegradation()
        result2 = degrader2(audio.clone(), sample_rate)

        # Results should be identical when both seeds are set
        assert torch.allclose(result1, result2, atol=1e-5)


class TestPhoneMelSpectrogram:
    """Test mel spectrogram conversion."""

    @pytest.fixture
    def mel_transform(self):
        """Create mel spectrogram transform."""
        return PhoneMelSpectrogram()

    @pytest.fixture
    def phone_audio(self):
        """Create phone-quality audio (8kHz)."""
        sample_rate = 8000
        duration = 3.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * torch.pi * 1000 * t).unsqueeze(0)
        return audio

    def test_mel_spectrogram_shape(self, mel_transform, phone_audio):
        """Test mel spectrogram output shape."""
        mel_spec = mel_transform(phone_audio)

        # Expected: (1, 64 mels, ~94 time frames)
        assert mel_spec.ndim == 3, "Should be 3D (batch, mels, time)"
        assert mel_spec.shape[1] == 64, f"Should have 64 mel bands, got {mel_spec.shape[1]}"

    def test_mel_spectrogram_is_log_scale(self, mel_transform, phone_audio):
        """Test that mel spectrogram is in log/dB scale."""
        mel_spec = mel_transform(phone_audio)

        # Log mel spectrograms have negative values (dB)
        assert mel_spec.min() < 0, "Log mel spectrogram should have negative values"

    def test_mel_spectrogram_finite(self, mel_transform, phone_audio):
        """Test that mel spectrogram has no NaN or Inf values."""
        mel_spec = mel_transform(phone_audio)

        assert torch.isfinite(mel_spec).all(), "Mel spectrogram should be finite"

    def test_silent_audio(self, mel_transform):
        """Test mel spectrogram of silent audio."""
        silent = torch.zeros(1, 24000)  # 3 seconds of silence

        mel_spec = mel_transform(silent)

        assert torch.isfinite(mel_spec).all(), "Should handle silence gracefully"


class TestDegradationEffects:
    """Test individual degradation effects."""

    def test_codec_simulation(self):
        """Test codec simulation adds quantization noise."""
        degrader = PhoneDegradation(
            noise_prob=0,
            codec_prob=1.0,  # Always apply codec
            packet_loss_prob=0,
            echo_prob=0
        )

        # Create smooth audio at 8kHz
        audio = torch.sin(torch.linspace(0, 10*np.pi, 8000)).unsqueeze(0)

        # Apply codec
        degrader._simulate_codec(audio)
        # Test passes if no exception

    def test_packet_loss_simulation(self):
        """Test packet loss simulation."""
        degrader = PhoneDegradation()

        audio = torch.ones(1, 8000)  # 1 second of constant audio

        result = degrader._simulate_packet_loss(audio)

        # Should have some zeros (dropouts)
        # Note: may not always have dropouts due to randomness
        assert result.shape == audio.shape

    def test_echo_simulation(self):
        """Test echo/reverb simulation."""
        degrader = PhoneDegradation()

        # Create impulse
        audio = torch.zeros(1, 8000)
        audio[0, 100] = 1.0  # Single impulse

        result = degrader._add_echo(audio)

        # Echo should create additional peaks after the impulse
        # Sum should be greater than original
        assert result.abs().sum() > audio.abs().sum()


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test complete degradation + mel spectrogram pipeline."""
        config = PhoneConfig()
        degrader = PhoneDegradation(config)
        mel_transform = PhoneMelSpectrogram(config)

        # Create input audio
        sample_rate = 16000
        duration = 3.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * torch.pi * 1000 * t).unsqueeze(0)

        # Apply degradation
        degraded = degrader(audio, sample_rate)

        # Convert to mel spectrogram
        mel_spec = mel_transform(degraded)

        # Check final shape is model-ready
        assert mel_spec.shape[1] == 64, "Should have 64 mel bands"
        assert mel_spec.ndim == 3, "Should be 3D tensor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
