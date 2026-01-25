"""
Unit tests for utility scripts.

Tests:
- test_detection.py functions
- degrade_to_phone.py functions
- create_mini_dataset.py utilities
"""

import pytest
import torch
import tempfile
import numpy as np
import soundfile as sf
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestDetectionScript:
    """Test detection script functions."""

    @pytest.fixture
    def test_audio_file(self):
        """Create a temporary test audio file."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Create 3 seconds of audio at 16kHz
            sample_rate = 16000
            duration = 3.0
            samples = int(sample_rate * duration)
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
            audio = audio.astype(np.float32)
            sf.write(f.name, audio, sample_rate)
            yield f.name
        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_preprocess_audio_returns_tensor(self, test_audio_file):
        """Test that preprocess_audio returns a tensor."""
        from test_detection import preprocess_audio
        from phone_augment import PhoneConfig, PhoneDegradation

        config = PhoneConfig()
        degrader = PhoneDegradation(config)

        waveform = preprocess_audio(test_audio_file, config, degrader)

        assert isinstance(waveform, torch.Tensor)
        assert waveform.ndim == 1  # 1D after squeeze

    def test_preprocess_audio_correct_sample_rate(self, test_audio_file):
        """Test that preprocessed audio is at phone sample rate."""
        from test_detection import preprocess_audio
        from phone_augment import PhoneConfig, PhoneDegradation

        config = PhoneConfig()
        degrader = PhoneDegradation(config)

        waveform = preprocess_audio(test_audio_file, config, degrader)

        # Should be approximately 3 seconds at 8kHz
        expected_samples = int(3.0 * 8000)
        # Allow some tolerance for resampling
        assert abs(len(waveform) - expected_samples) < 100

    def test_detect_deepfake_returns_result(self, test_audio_file):
        """Test that detect_deepfake returns a result dictionary."""
        from test_detection import detect_deepfake
        from phone_augment import PhoneConfig, PhoneDegradation, PhoneMelSpectrogram
        import torch.nn as nn

        config = PhoneConfig()
        degrader = PhoneDegradation(config)
        mel_transform = PhoneMelSpectrogram(config)

        # Create dummy model
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 94, 2)
        )
        model.eval()

        result = detect_deepfake(
            test_audio_file,
            model,
            mel_transform,
            degrader,
            config,
            device='cpu',
            threshold=0.5
        )

        assert 'prediction' in result
        assert 'confidence' in result
        assert 'real_probability' in result
        assert 'fake_probability' in result
        assert result['prediction'] in ['REAL', 'FAKE']

    def test_detect_deepfake_threshold(self, test_audio_file):
        """Test that threshold affects prediction."""
        from test_detection import detect_deepfake
        from phone_augment import PhoneConfig, PhoneDegradation, PhoneMelSpectrogram
        import torch.nn as nn

        config = PhoneConfig()
        degrader = PhoneDegradation(config)
        mel_transform = PhoneMelSpectrogram(config)

        # Create model that outputs fixed probabilities
        class FixedModel(nn.Module):
            def forward(self, x):
                # Output that gives ~60% fake probability
                return torch.tensor([[0.4, 0.6]])

        model = FixedModel()
        model.eval()

        # With 0.5 threshold, should be FAKE
        result1 = detect_deepfake(
            test_audio_file, model, mel_transform, degrader, config,
            device='cpu', threshold=0.5
        )
        assert result1['prediction'] == 'FAKE'

        # With 0.7 threshold, should be REAL
        result2 = detect_deepfake(
            test_audio_file, model, mel_transform, degrader, config,
            device='cpu', threshold=0.7
        )
        assert result2['prediction'] == 'REAL'


class TestLoadModel:
    """Test model loading function."""

    def test_load_model_creates_model(self):
        """Test that load_model creates a model from checkpoint."""
        from test_detection import load_model
        from model import get_model
        import torch.nn as nn

        # Create a checkpoint
        model = get_model('mobile')

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': 10,
                'val_accuracy': 0.85
            }, f.name)

            loaded_model = load_model(f.name, 'mobile', 'cpu')

            assert loaded_model is not None
            assert isinstance(loaded_model, nn.Module)

            Path(f.name).unlink()


class TestDegradeScript:
    """Test degrade_to_phone.py functions."""

    @pytest.fixture
    def input_dir(self):
        """Create temporary input directory with audio files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test audio files
            sample_rate = 16000
            duration = 1.0
            samples = int(sample_rate * duration)

            for i in range(3):
                audio = np.random.randn(samples).astype(np.float32) * 0.5
                sf.write(str(Path(tmpdir) / f"test_{i}.wav"), audio, sample_rate)

            yield tmpdir

    def test_degrade_directory_creates_output(self, input_dir):
        """Test that degrade_directory creates output files."""
        from degrade_to_phone import degrade_directory

        with tempfile.TemporaryDirectory() as output_dir:
            degrade_directory(input_dir, output_dir)

            output_files = list(Path(output_dir).glob("*.wav"))
            assert len(output_files) == 3

    def test_degrade_directory_maintains_structure(self, input_dir):
        """Test that directory structure is maintained."""
        from degrade_to_phone import degrade_directory

        # Create subdirectory with files
        subdir = Path(input_dir) / "subdir"
        subdir.mkdir()

        audio = np.random.randn(8000).astype(np.float32)
        sf.write(str(subdir / "nested.wav"), audio, 16000)

        with tempfile.TemporaryDirectory() as output_dir:
            degrade_directory(input_dir, output_dir)

            # Check nested file exists
            nested_output = Path(output_dir) / "subdir" / "nested.wav"
            assert nested_output.exists()

    def test_degraded_audio_is_8khz(self, input_dir):
        """Test that degraded audio is at 8kHz."""
        from degrade_to_phone import degrade_directory

        with tempfile.TemporaryDirectory() as output_dir:
            degrade_directory(input_dir, output_dir)

            output_files = list(Path(output_dir).glob("*.wav"))
            assert len(output_files) > 0

            audio, sr = sf.read(str(output_files[0]))
            assert sr == 8000


class TestPrintResult:
    """Test result printing function."""

    def test_print_result_fake(self, capsys):
        """Test print_result for fake detection."""
        from test_detection import print_result

        result = {
            'file': 'test.wav',
            'prediction': 'FAKE',
            'confidence': 0.95,
            'real_probability': 0.05,
            'fake_probability': 0.95,
            'threshold': 0.5
        }

        print_result(result)
        captured = capsys.readouterr()

        assert 'FAKE' in captured.out
        assert '95' in captured.out  # Confidence percentage

    def test_print_result_real(self, capsys):
        """Test print_result for real detection."""
        from test_detection import print_result

        result = {
            'file': 'test.wav',
            'prediction': 'REAL',
            'confidence': 0.90,
            'real_probability': 0.90,
            'fake_probability': 0.10,
            'threshold': 0.5
        }

        print_result(result)
        captured = capsys.readouterr()

        assert 'REAL' in captured.out

    def test_print_result_verbose(self, capsys):
        """Test verbose output."""
        from test_detection import print_result

        result = {
            'file': 'test.wav',
            'prediction': 'FAKE',
            'confidence': 0.85,
            'real_probability': 0.15,
            'fake_probability': 0.85,
            'threshold': 0.5
        }

        print_result(result, verbose=True)
        captured = capsys.readouterr()

        assert 'Real:' in captured.out
        assert 'Fake:' in captured.out
        assert 'Threshold:' in captured.out


class TestAudioFormats:
    """Test handling of different audio formats."""

    def test_stereo_to_mono_conversion(self):
        """Test that stereo audio is converted to mono."""
        from test_detection import preprocess_audio
        from phone_augment import PhoneConfig, PhoneDegradation

        config = PhoneConfig()
        degrader = PhoneDegradation(config)

        # Create stereo audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sample_rate = 16000
            duration = 1.0
            samples = int(sample_rate * duration)
            # Stereo: 2 channels
            audio = np.random.randn(samples, 2).astype(np.float32)
            sf.write(f.name, audio, sample_rate)

            waveform = preprocess_audio(f.name, config, degrader)

            # Should be 1D (mono)
            assert waveform.ndim == 1

            Path(f.name).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
