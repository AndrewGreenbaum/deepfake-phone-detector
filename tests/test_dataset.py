"""
Unit tests for dataset loading and preprocessing.

Tests:
- PhoneAudioDataset functionality
- PreDegradedDataset functionality
- Dataloader creation
- Audio preprocessing utilities
"""

import pytest
import torch
import tempfile
import os
import numpy as np
import soundfile as sf
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from phone_augment import PhoneConfig


class TestPhoneConfigForDataset:
    """Test PhoneConfig integration with dataset."""

    def test_num_samples_calculation(self):
        """Test correct calculation of number of samples."""
        config = PhoneConfig()
        # 3 seconds at 8000 Hz = 24000 samples
        assert config.NUM_SAMPLES == 24000


class TestDatasetUtilities:
    """Test dataset utility functions."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create temporary dataset directory with dummy audio files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory structure
            real_dir = Path(tmpdir) / "real"
            fake_dir = Path(tmpdir) / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()

            # Create dummy WAV files (8kHz, 3 seconds)
            sample_rate = 8000
            duration = 3.0
            samples = int(sample_rate * duration)

            # Create real samples
            for i in range(5):
                audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
                audio = audio.astype(np.float32)
                sf.write(str(real_dir / f"real_{i}.wav"), audio, sample_rate)

            # Create fake samples
            for i in range(5):
                audio = np.sin(2 * np.pi * 880 * np.linspace(0, duration, samples))
                audio = audio.astype(np.float32)
                sf.write(str(fake_dir / f"fake_{i}.wav"), audio, sample_rate)

            yield tmpdir

    def test_directory_structure_loading(self, temp_dataset_dir):
        """Test that dataset correctly loads from directory structure."""
        from dataset import PhoneAudioDataset

        dataset = PhoneAudioDataset(
            temp_dataset_dir,
            apply_degradation=False,
            augment=False
        )

        assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"

    def test_labels_assigned_correctly(self, temp_dataset_dir):
        """Test that labels are correctly assigned (0=real, 1=fake)."""
        from dataset import PhoneAudioDataset

        dataset = PhoneAudioDataset(
            temp_dataset_dir,
            apply_degradation=False,
            augment=False
        )

        # Check labels
        labels = [dataset.samples[i][1] for i in range(len(dataset))]
        assert labels.count(0) == 5, "Should have 5 real samples (label=0)"
        assert labels.count(1) == 5, "Should have 5 fake samples (label=1)"

    def test_getitem_returns_correct_shape(self, temp_dataset_dir):
        """Test that __getitem__ returns correct tensor shapes."""
        from dataset import PhoneAudioDataset

        dataset = PhoneAudioDataset(
            temp_dataset_dir,
            apply_degradation=False,
            augment=False
        )

        spec, label = dataset[0]

        # Mel spectrogram shape: (1, n_mels, time_frames)
        assert spec.ndim == 3, f"Expected 3D tensor, got {spec.ndim}D"
        assert spec.shape[0] == 1, "Channel dimension should be 1"
        assert spec.shape[1] == 64, f"Expected 64 mel bands, got {spec.shape[1]}"

    def test_getitem_returns_valid_label(self, temp_dataset_dir):
        """Test that labels are 0 or 1."""
        from dataset import PhoneAudioDataset

        dataset = PhoneAudioDataset(
            temp_dataset_dir,
            apply_degradation=False,
            augment=False
        )

        for i in range(len(dataset)):
            _, label = dataset[i]
            assert label in [0, 1], f"Label should be 0 or 1, got {label}"


class TestPadOrTruncate:
    """Test audio padding and truncation."""

    def test_short_audio_is_padded(self):
        """Test that short audio is zero-padded."""
        from dataset import PhoneAudioDataset

        config = PhoneConfig()

        # Create a mock dataset to access the method
        with patch.object(PhoneAudioDataset, '__init__', lambda x, *args, **kwargs: None):
            dataset = PhoneAudioDataset.__new__(PhoneAudioDataset)
            dataset.config = config
            dataset.augment = False

            # Short audio (1 second = 8000 samples)
            short_audio = torch.randn(1, 8000)
            padded = dataset._pad_or_truncate(short_audio)

            assert padded.shape[1] == config.NUM_SAMPLES
            # Last portion should be zeros
            assert torch.allclose(padded[0, 8000:], torch.zeros(config.NUM_SAMPLES - 8000))

    def test_long_audio_is_truncated(self):
        """Test that long audio is truncated."""
        from dataset import PhoneAudioDataset

        config = PhoneConfig()

        with patch.object(PhoneAudioDataset, '__init__', lambda x, *args, **kwargs: None):
            dataset = PhoneAudioDataset.__new__(PhoneAudioDataset)
            dataset.config = config
            dataset.augment = False

            # Long audio (5 seconds = 40000 samples)
            long_audio = torch.randn(1, 40000)
            truncated = dataset._pad_or_truncate(long_audio)

            assert truncated.shape[1] == config.NUM_SAMPLES

    def test_exact_length_unchanged(self):
        """Test that exact-length audio is unchanged."""
        from dataset import PhoneAudioDataset

        config = PhoneConfig()

        with patch.object(PhoneAudioDataset, '__init__', lambda x, *args, **kwargs: None):
            dataset = PhoneAudioDataset.__new__(PhoneAudioDataset)
            dataset.config = config
            dataset.augment = False

            exact_audio = torch.randn(1, config.NUM_SAMPLES)
            result = dataset._pad_or_truncate(exact_audio)

            assert result.shape[1] == config.NUM_SAMPLES
            assert torch.allclose(result, exact_audio)


class TestPreDegradedDataset:
    """Test PreDegradedDataset class."""

    @pytest.fixture
    def pre_degraded_dir(self):
        """Create temporary pre-degraded dataset directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir) / "real"
            fake_dir = Path(tmpdir) / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()

            # Create 8kHz audio files (pre-degraded format)
            sample_rate = 8000
            duration = 3.0
            samples = int(sample_rate * duration)

            for i in range(3):
                audio = np.random.randn(samples).astype(np.float32) * 0.5
                sf.write(str(real_dir / f"real_{i}.wav"), audio, sample_rate)
                sf.write(str(fake_dir / f"fake_{i}.wav"), audio, sample_rate)

            yield tmpdir

    def test_pre_degraded_loading(self, pre_degraded_dir):
        """Test loading pre-degraded dataset."""
        from dataset import PreDegradedDataset

        dataset = PreDegradedDataset(pre_degraded_dir)
        assert len(dataset) == 6

    def test_pre_degraded_output_shape(self, pre_degraded_dir):
        """Test output shape from pre-degraded dataset."""
        from dataset import PreDegradedDataset

        dataset = PreDegradedDataset(pre_degraded_dir)
        spec, label = dataset[0]

        assert spec.ndim == 3
        assert spec.shape[1] == 64  # n_mels


class TestCreateDataloaders:
    """Test dataloader creation function."""

    @pytest.fixture
    def dataset_dir(self):
        """Create dataset directory for dataloader testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir) / "real"
            fake_dir = Path(tmpdir) / "fake"
            real_dir.mkdir()
            fake_dir.mkdir()

            sample_rate = 8000
            samples = 24000  # 3 seconds

            # Create enough samples for splits
            for i in range(20):
                audio = np.random.randn(samples).astype(np.float32) * 0.5
                sf.write(str(real_dir / f"real_{i}.wav"), audio, sample_rate)
                sf.write(str(fake_dir / f"fake_{i}.wav"), audio, sample_rate)

            yield tmpdir

    def test_dataloader_creation(self, dataset_dir):
        """Test that dataloaders are created correctly."""
        from dataset import create_dataloaders

        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_dir,
            batch_size=4,
            num_workers=0,  # Use 0 workers for testing
            apply_degradation=False
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_dataloader_splits(self, dataset_dir):
        """Test that data is split correctly."""
        from dataset import create_dataloaders

        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_dir,
            batch_size=4,
            train_split=0.8,
            val_split=0.1,
            num_workers=0,
            apply_degradation=False
        )

        total_samples = 40  # 20 real + 20 fake
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        test_size = len(test_loader.dataset)

        assert train_size + val_size + test_size == total_samples
        assert train_size == int(total_samples * 0.8)

    def test_dataloader_batch_shape(self, dataset_dir):
        """Test that batches have correct shape."""
        from dataset import create_dataloaders

        train_loader, _, _ = create_dataloaders(
            dataset_dir,
            batch_size=4,
            num_workers=0,
            apply_degradation=False
        )

        batch = next(iter(train_loader))
        inputs, labels = batch

        assert inputs.shape[0] <= 4  # Batch size
        assert inputs.ndim == 4  # (batch, channel, mels, time)
        assert labels.shape[0] == inputs.shape[0]


class TestErrorHandling:
    """Test error handling in dataset."""

    def test_empty_directory(self):
        """Test behavior with empty directory."""
        from dataset import PhoneAudioDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = PhoneAudioDataset(
                tmpdir,
                apply_degradation=False
            )
            assert len(dataset) == 0

    def test_corrupted_file_handling(self):
        """Test that corrupted files don't crash the dataset."""
        from dataset import PhoneAudioDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir) / "real"
            real_dir.mkdir()

            # Create a corrupted "wav" file
            with open(real_dir / "corrupted.wav", "wb") as f:
                f.write(b"not a valid wav file")

            # Also create a valid file
            audio = np.random.randn(24000).astype(np.float32)
            sf.write(str(real_dir / "valid.wav"), audio, 8000)

            dataset = PhoneAudioDataset(
                tmpdir,
                apply_degradation=False
            )

            # Should load both files
            assert len(dataset) == 2

            # Accessing corrupted file should return dummy tensor
            # (implementation returns zeros on error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
