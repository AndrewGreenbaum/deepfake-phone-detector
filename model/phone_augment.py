"""
Phone Call Audio Degradation Pipeline

Simulates the audio quality of real phone calls:
- GSM/AMR codec compression
- Telephone bandpass filter (300Hz - 3.4kHz)
- 8kHz sample rate
- Background noise
- Packet loss simulation
- Echo and reverb
"""

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from scipy import signal
from scipy.io import wavfile
import random
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class PhoneConfig:
    """Phone audio configuration matching real telephony."""
    SAMPLE_RATE = 8000          # Standard phone sample rate
    LOW_FREQ = 300              # Phone low cutoff
    HIGH_FREQ = 3400            # Phone high cutoff
    N_MELS = 64                 # Fewer mels for narrow bandwidth
    N_FFT = 512                 # Smaller FFT for 8kHz
    HOP_LENGTH = 256
    DURATION = 3.0              # Seconds per sample
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)


class PhoneDegradation:
    """
    Transforms clean audio into phone-quality audio.

    This is the key innovation - we train on audio that sounds
    like actual phone calls, not studio recordings.
    """

    def __init__(
        self,
        config: PhoneConfig = PhoneConfig(),
        noise_prob: float = 0.7,
        codec_prob: float = 0.9,
        packet_loss_prob: float = 0.3,
        echo_prob: float = 0.2
    ):
        self.config = config
        self.noise_prob = noise_prob
        self.codec_prob = codec_prob
        self.packet_loss_prob = packet_loss_prob
        self.echo_prob = echo_prob

        # Pre-compute bandpass filter coefficients
        self.b, self.a = signal.butter(
            4,
            [config.LOW_FREQ, config.HIGH_FREQ],
            btype='band',
            fs=config.SAMPLE_RATE
        )

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply full phone degradation pipeline."""

        # 1. Resample to phone rate (8kHz)
        if sample_rate != self.config.SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.config.SAMPLE_RATE
            )

        # 2. Apply telephone bandpass filter
        waveform = self._apply_bandpass(waveform)

        # 3. Simulate codec compression (GSM-like artifacts)
        if random.random() < self.codec_prob:
            waveform = self._simulate_codec(waveform)

        # 4. Add background noise
        if random.random() < self.noise_prob:
            waveform = self._add_noise(waveform)

        # 5. Simulate packet loss
        if random.random() < self.packet_loss_prob:
            waveform = self._simulate_packet_loss(waveform)

        # 6. Add echo
        if random.random() < self.echo_prob:
            waveform = self._add_echo(waveform)

        # 7. Normalize
        waveform = self._normalize(waveform)

        return waveform

    def _apply_bandpass(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply telephone bandpass filter (300Hz - 3.4kHz)."""
        # Convert to numpy for scipy filter
        audio_np = waveform.numpy().squeeze()

        # Apply filter
        filtered = signal.filtfilt(self.b, self.a, audio_np)

        return torch.tensor(filtered, dtype=torch.float32).unsqueeze(0)

    def _simulate_codec(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Simulate GSM/AMR codec compression artifacts.

        Real codecs use complex CELP encoding. We approximate with:
        - Quantization noise
        - Slight spectral smoothing
        - Frame-based processing artifacts
        """
        audio = waveform.numpy().squeeze()

        # Frame-based processing (GSM uses 20ms frames)
        frame_size = int(0.02 * self.config.SAMPLE_RATE)  # 20ms
        num_frames = len(audio) // frame_size

        processed = np.zeros_like(audio)

        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio[start:end]

            # Quantization (GSM uses ~13-bit effective)
            bits = random.randint(10, 13)
            levels = 2 ** bits
            frame = np.round(frame * levels) / levels

            # Add tiny frame-boundary artifacts
            if i > 0 and random.random() < 0.1:
                frame[:5] *= 0.95  # Slight discontinuity

            processed[start:end] = frame

        return torch.tensor(processed, dtype=torch.float32).unsqueeze(0)

    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add realistic phone call background noise."""
        audio = waveform.numpy().squeeze()

        noise_type = random.choice(['white', 'pink', 'hum', 'ambient'])

        if noise_type == 'white':
            noise = np.random.randn(len(audio)) * random.uniform(0.002, 0.01)

        elif noise_type == 'pink':
            # Pink noise (1/f)
            white = np.random.randn(len(audio))
            # Simple approximation
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            noise = signal.lfilter(b, a, white) * random.uniform(0.005, 0.015)

        elif noise_type == 'hum':
            # 60Hz electrical hum (common in phone lines)
            t = np.arange(len(audio)) / self.config.SAMPLE_RATE
            hum = np.sin(2 * np.pi * 60 * t) * random.uniform(0.001, 0.005)
            # Add harmonics
            hum += np.sin(2 * np.pi * 120 * t) * random.uniform(0.0005, 0.002)
            noise = hum

        else:  # ambient
            # Low-frequency rumble
            noise = np.random.randn(len(audio))
            b, a = signal.butter(2, 500, btype='low', fs=self.config.SAMPLE_RATE)
            noise = signal.filtfilt(b, a, noise) * random.uniform(0.005, 0.02)

        return torch.tensor(audio + noise, dtype=torch.float32).unsqueeze(0)

    def _simulate_packet_loss(self, waveform: torch.Tensor) -> torch.Tensor:
        """Simulate VoIP packet loss (audio dropouts)."""
        audio = waveform.numpy().squeeze()

        # Packet size (~20ms)
        packet_size = int(0.02 * self.config.SAMPLE_RATE)
        num_packets = len(audio) // packet_size

        # Lose 1-5% of packets
        loss_rate = random.uniform(0.01, 0.05)

        for i in range(num_packets):
            if random.random() < loss_rate:
                start = i * packet_size
                end = start + packet_size

                # Replace with silence or interpolation
                if random.random() < 0.5:
                    audio[start:end] = 0  # Silence
                else:
                    # Simple interpolation (packet loss concealment)
                    if i > 0:
                        prev_end = audio[start-10:start].mean() if start >= 10 else 0
                        audio[start:end] = np.linspace(prev_end, 0, packet_size)

        return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    def _add_echo(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add telephone echo/reverb."""
        audio = waveform.numpy().squeeze()

        # Echo delay (50-200ms typical for phone)
        delay_ms = random.uniform(50, 200)
        delay_samples = int(delay_ms * self.config.SAMPLE_RATE / 1000)

        # Echo strength
        echo_strength = random.uniform(0.1, 0.3)

        # Create echo
        echo = np.zeros_like(audio)
        if delay_samples < len(audio):
            echo[delay_samples:] = audio[:-delay_samples] * echo_strength

        return torch.tensor(audio + echo, dtype=torch.float32).unsqueeze(0)

    def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val * 0.95
        return waveform


class PhoneMelSpectrogram:
    """
    Convert phone audio to mel spectrogram.
    Optimized for 8kHz, narrow bandwidth.
    """

    def __init__(self, config: PhoneConfig = PhoneConfig()):
        self.config = config

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.LOW_FREQ,
            f_max=config.HIGH_FREQ
        )

        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to log mel spectrogram."""
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.db_transform(mel_spec)
        return log_mel_spec


def degrade_audio_file(
    input_path: str,
    output_path: str,
    degrader: Optional[PhoneDegradation] = None
):
    """Degrade a single audio file to phone quality."""
    if degrader is None:
        degrader = PhoneDegradation()

    # Load audio
    waveform, sample_rate = torchaudio.load(input_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Apply degradation
    degraded = degrader(waveform, sample_rate)

    # Save
    torchaudio.save(output_path, degraded, degrader.config.SAMPLE_RATE)


if __name__ == "__main__":
    # Test the degradation pipeline
    print("Testing phone degradation pipeline...")

    config = PhoneConfig()
    degrader = PhoneDegradation(config)
    mel_transform = PhoneMelSpectrogram(config)

    # Create test audio (1kHz tone)
    duration = 3.0
    sample_rate = 16000
    t = torch.linspace(0, duration, int(sample_rate * duration))
    test_audio = torch.sin(2 * torch.pi * 1000 * t).unsqueeze(0)

    print(f"Input: {test_audio.shape}, {sample_rate}Hz")

    # Degrade
    degraded = degrader(test_audio, sample_rate)
    print(f"Degraded: {degraded.shape}, {config.SAMPLE_RATE}Hz")

    # Convert to spectrogram
    mel_spec = mel_transform(degraded)
    print(f"Mel spectrogram: {mel_spec.shape}")

    print("\nPhone degradation pipeline working!")
