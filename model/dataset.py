"""
Dataset loader for phone-quality deepfake detection.
Handles 8kHz audio with phone degradation augmentation.
"""

import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
from pathlib import Path

from phone_augment import PhoneConfig, PhoneDegradation, PhoneMelSpectrogram


class PhoneAudioDataset(Dataset):
    """
    Dataset for phone-quality deepfake detection.

    Features:
    - Loads audio and applies phone degradation
    - Outputs 8kHz mel spectrograms
    - Supports both directory structure and CSV labels
    """

    def __init__(
        self,
        data_path: str,
        csv_file: Optional[str] = None,
        config: PhoneConfig = PhoneConfig(),
        apply_degradation: bool = True,
        augment: bool = True
    ):
        self.config = config
        self.apply_degradation = apply_degradation
        self.augment = augment

        # Initialize transforms
        self.degrader = PhoneDegradation(config) if apply_degradation else None
        self.mel_transform = PhoneMelSpectrogram(config)

        # Load samples
        if csv_file and os.path.exists(csv_file):
            self.samples = self._load_from_csv(csv_file)
        else:
            self.samples = self._load_from_directory(data_path)

        print(f"Loaded {len(self.samples)} samples")
        real_count = sum(1 for _, l in self.samples if l == 0)
        fake_count = sum(1 for _, l in self.samples if l == 1)
        print(f"  Real: {real_count}, Fake: {fake_count}")

    def _load_from_csv(self, csv_file: str) -> List[Tuple[str, int]]:
        """Load samples from CSV file."""
        df = pd.read_csv(csv_file)
        return [(row['filepath'], row['label']) for _, row in df.iterrows()]

    def _load_from_directory(self, data_path: str) -> List[Tuple[str, int]]:
        """Load samples from directory structure (real/ and fake/ folders)."""
        samples = []
        data_path = Path(data_path)

        # Real samples (label = 0)
        real_dir = data_path / 'real'
        if real_dir.exists():
            for f in real_dir.iterdir():
                if f.suffix in ['.wav', '.flac', '.mp3', '.ogg']:
                    samples.append((str(f), 0))

        # Fake samples (label = 1)
        fake_dir = data_path / 'fake'
        if fake_dir.exists():
            for f in fake_dir.iterdir():
                if f.suffix in ['.wav', '.flac', '.mp3', '.ogg']:
                    samples.append((str(f), 1))

        return samples

    def _load_audio(self, filepath: str) -> Tuple[torch.Tensor, int]:
        """Load audio file."""
        waveform, sr = torchaudio.load(filepath)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform, sr

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Ensure audio is exactly the right length."""
        target_length = self.config.NUM_SAMPLES

        if waveform.shape[1] < target_length:
            # Pad with zeros
            padding = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Random crop if augmenting, else take from start
            if self.augment and waveform.shape[1] > target_length:
                start = torch.randint(0, waveform.shape[1] - target_length, (1,)).item()
            else:
                start = 0
            waveform = waveform[:, start:start + target_length]

        return waveform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]

        try:
            # Load audio
            waveform, sr = self._load_audio(filepath)

            # Apply phone degradation (includes resampling to 8kHz)
            if self.degrader and self.apply_degradation:
                waveform = self.degrader(waveform, sr)
            elif sr != self.config.SAMPLE_RATE:
                # Just resample if no degradation
                waveform = torchaudio.functional.resample(
                    waveform, sr, self.config.SAMPLE_RATE
                )

            # Pad or truncate to fixed length
            waveform = self._pad_or_truncate(waveform)

            # Convert to mel spectrogram
            mel_spec = self.mel_transform(waveform)

            return mel_spec, label

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return zero tensor on error
            dummy_spec = torch.zeros(1, self.config.N_MELS, 94)
            return dummy_spec, label


class PreDegradedDataset(Dataset):
    """
    Dataset for pre-degraded phone audio.
    Use this when audio is already phone-quality (faster training).
    """

    def __init__(
        self,
        data_path: str,
        config: PhoneConfig = PhoneConfig()
    ):
        self.config = config
        self.mel_transform = PhoneMelSpectrogram(config)
        self.samples = self._load_from_directory(data_path)

        print(f"Loaded {len(self.samples)} pre-degraded samples")

    def _load_from_directory(self, data_path: str) -> List[Tuple[str, int]]:
        """Load samples from directory."""
        samples = []
        data_path = Path(data_path)

        for label_name, label in [('real', 0), ('fake', 1)]:
            label_dir = data_path / label_name
            if label_dir.exists():
                for f in label_dir.iterdir():
                    if f.suffix in ['.wav', '.flac']:
                        samples.append((str(f), label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, label = self.samples[idx]

        # Use soundfile instead of torchaudio for better FFmpeg compatibility
        audio, sr = sf.read(filepath)

        # Convert to float32 and ensure contiguous
        audio = np.ascontiguousarray(audio.astype(np.float32))

        # Handle stereo -> mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Convert to tensor with channel dimension
        waveform = torch.from_numpy(audio).unsqueeze(0)

        # Should already be 8kHz, but resample if needed
        if sr != self.config.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.config.SAMPLE_RATE
            )

        # Fixed length
        target = self.config.NUM_SAMPLES
        if waveform.shape[1] < target:
            waveform = torch.nn.functional.pad(waveform, (0, target - waveform.shape[1]))
        else:
            waveform = waveform[:, :target]

        mel_spec = self.mel_transform(waveform)
        return mel_spec, label


def create_dataloaders(
    data_path: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 4,
    apply_degradation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""

    config = PhoneConfig()

    # Create dataset
    full_dataset = PhoneAudioDataset(
        data_path,
        config=config,
        apply_degradation=apply_degradation,
        augment=True
    )

    # Split
    total = len(full_dataset)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Disable augmentation for val/test
    # (Note: this is a simplification - ideally use separate dataset instances)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    config = PhoneConfig()
    print(f"Phone config: {config.SAMPLE_RATE}Hz, {config.N_MELS} mels")

    # Test with dummy data
    dataset = PhoneAudioDataset("../data", config=config, apply_degradation=True)

    if len(dataset) > 0:
        spec, label = dataset[0]
        print(f"Spectrogram shape: {spec.shape}")
        print(f"Label: {'Fake' if label == 1 else 'Real'}")
