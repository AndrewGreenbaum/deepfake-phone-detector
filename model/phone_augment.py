"""
Phone Call Audio Degradation Pipeline

Simulates the audio quality of real phone calls:
- GSM/AMR codec compression
- Telephone bandpass filter (300Hz - 3.4kHz)
- 8kHz sample rate
- Background noise
- Packet loss simulation (with burst model)
- Echo and reverb
- VAD/DTX onset clipping and comfort noise
- Spectral smoothing (codec-style)
- Soft clipping (microphone saturation)
- AGC (automatic gain control)
- Jitter (timing variations)
- Codec harmonic distortion
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
        echo_prob: float = 0.2,
        # New parameters for enhanced realism
        vad_prob: float = 0.5,
        clipping_prob: float = 0.3,
        agc_prob: float = 0.6,
        jitter_prob: float = 0.4,
        spectral_smoothing_prob: float = 0.7,
        codec_distortion_prob: float = 0.5,
    ):
        self.config = config
        self.noise_prob = noise_prob
        self.codec_prob = codec_prob
        self.packet_loss_prob = packet_loss_prob
        self.echo_prob = echo_prob
        self.vad_prob = vad_prob
        self.clipping_prob = clipping_prob
        self.agc_prob = agc_prob
        self.jitter_prob = jitter_prob
        self.spectral_smoothing_prob = spectral_smoothing_prob
        self.codec_distortion_prob = codec_distortion_prob

        # Pre-compute bandpass filter coefficients
        self.b, self.a = signal.butter(
            4,
            [config.LOW_FREQ, config.HIGH_FREQ],
            btype='band',
            fs=config.SAMPLE_RATE
        )

        # Pre-compute spectral smoothing lowpass filter (above 2kHz)
        self.smooth_b, self.smooth_a = signal.butter(
            2,
            2000,
            btype='low',
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

        # 3. Apply spectral smoothing (codec-style high-freq loss)
        if random.random() < self.spectral_smoothing_prob:
            waveform = self._apply_spectral_smoothing(waveform)

        # 4. Simulate codec compression (GSM-like artifacts)
        if random.random() < self.codec_prob:
            waveform = self._simulate_codec(waveform)

        # 5. Apply codec distortion (harmonic distortion + resonance)
        if random.random() < self.codec_distortion_prob:
            waveform = self._apply_codec_distortion(waveform)

        # 6. Apply AGC (automatic gain control)
        if random.random() < self.agc_prob:
            waveform = self._apply_agc(waveform)

        # 7. Apply soft clipping (microphone saturation)
        if random.random() < self.clipping_prob:
            waveform = self._apply_clipping(waveform)

        # 8. Add background noise
        if random.random() < self.noise_prob:
            waveform = self._add_noise(waveform)

        # 9. Simulate burst packet loss
        if random.random() < self.packet_loss_prob:
            waveform = self._simulate_burst_packet_loss(waveform)

        # 10. Apply jitter (timing variations)
        if random.random() < self.jitter_prob:
            waveform = self._apply_jitter(waveform)

        # 11. Simulate VAD/DTX (onset clipping + comfort noise)
        if random.random() < self.vad_prob:
            waveform = self._simulate_vad_dtx(waveform)

        # 12. Add echo
        if random.random() < self.echo_prob:
            waveform = self._add_echo(waveform)

        # 13. Normalize
        waveform = self._normalize(waveform)

        return waveform

    def _apply_bandpass(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply telephone bandpass filter (300Hz - 3.4kHz)."""
        # Convert to numpy for scipy filter
        audio_np = waveform.numpy().squeeze()

        # Apply filter
        filtered = signal.filtfilt(self.b, self.a, audio_np)

        # Use .copy() to ensure contiguous array with positive strides
        return torch.tensor(np.ascontiguousarray(filtered), dtype=torch.float32).unsqueeze(0)

    def _apply_spectral_smoothing(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Simulate codec spectral smoothing.

        Codecs use LPC-based encoding which smooths spectral detail,
        especially in higher frequencies. This reduces fricative clarity.
        """
        audio = waveform.numpy().squeeze()

        # Blend original with lowpass-filtered version for spectral smoothing
        # This simulates LPC-based codec smoothing effect
        smoothed = signal.filtfilt(self.smooth_b, self.smooth_a, audio)

        # Random blend amount (more smoothing = more codec-like)
        blend = random.uniform(0.2, 0.5)
        result = audio * (1 - blend) + smoothed * blend

        return torch.tensor(np.ascontiguousarray(result), dtype=torch.float32).unsqueeze(0)

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

    def _apply_codec_distortion(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add harmonic distortion and resonance typical of low-bitrate codecs.

        Creates the "metallic" or "buzzy" quality heard on poor phone connections.
        """
        audio = waveform.numpy().squeeze()

        # Add subtle harmonic distortion (2nd and 3rd harmonics)
        # This simulates non-linear codec behavior
        distortion_amount = random.uniform(0.02, 0.08)

        # Soft harmonic generation
        harmonics = (
            distortion_amount * 0.6 * np.sign(audio) * audio ** 2 +  # 2nd harmonic
            distortion_amount * 0.3 * audio ** 3  # 3rd harmonic
        )

        # Add slight resonance at common formant frequencies (500Hz, 1500Hz, 2500Hz)
        resonance_strength = random.uniform(0.01, 0.04)
        for freq in [500, 1500, 2500]:
            if random.random() < 0.5:  # Randomly apply each resonance
                # Create narrow bandpass at formant frequency
                bw = 100  # Bandwidth
                low = max(freq - bw/2, 100)
                high = min(freq + bw/2, self.config.HIGH_FREQ)
                b_res, a_res = signal.butter(2, [low, high], btype='band', fs=self.config.SAMPLE_RATE)
                resonant = signal.filtfilt(b_res, a_res, audio)
                harmonics += resonant * resonance_strength

        result = audio + harmonics

        return torch.tensor(np.ascontiguousarray(result), dtype=torch.float32).unsqueeze(0)

    def _apply_agc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply automatic gain control (AGC) compression.

        Phones use AGC to normalize volume levels. This compresses
        dynamic range with attack/release characteristics.
        """
        audio = waveform.numpy().squeeze()

        # AGC parameters
        attack_time = random.uniform(0.005, 0.02)  # 5-20ms attack
        release_time = random.uniform(0.05, 0.2)   # 50-200ms release
        target_level = random.uniform(0.3, 0.5)
        ratio = random.uniform(2.0, 4.0)           # Compression ratio 2:1 to 4:1

        # Convert times to coefficients
        attack_coef = np.exp(-1.0 / (attack_time * self.config.SAMPLE_RATE))
        release_coef = np.exp(-1.0 / (release_time * self.config.SAMPLE_RATE))

        # Process sample by sample
        envelope = 0.0
        result = np.zeros_like(audio)

        for i in range(len(audio)):
            input_level = abs(audio[i])

            # Envelope follower with attack/release
            if input_level > envelope:
                envelope = attack_coef * envelope + (1 - attack_coef) * input_level
            else:
                envelope = release_coef * envelope + (1 - release_coef) * input_level

            # Compute gain based on compression ratio
            if envelope > target_level:
                gain = target_level + (envelope - target_level) / ratio
                gain = gain / (envelope + 1e-10)
            else:
                gain = 1.0

            result[i] = audio[i] * gain

        return torch.tensor(np.ascontiguousarray(result), dtype=torch.float32).unsqueeze(0)

    def _apply_clipping(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply soft clipping for microphone saturation on loud sounds.

        Uses tanh for smooth saturation curve, simulating analog limiting.
        """
        threshold = random.uniform(0.7, 0.9)

        # Soft clip using tanh (smooth saturation)
        clipped = torch.tanh(waveform / threshold) * threshold

        return clipped

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

    def _simulate_burst_packet_loss(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Simulate VoIP packet loss with Markov model for bursty behavior.

        Real packet loss is bursty - when one packet is lost, the next
        is more likely to be lost too. This creates realistic dropout patterns.
        """
        audio = waveform.numpy().squeeze()

        # Packet size (~20ms)
        packet_size = int(0.02 * self.config.SAMPLE_RATE)
        num_packets = len(audio) // packet_size

        # Markov model probabilities
        # P(loss | no_loss) - probability to start losing
        p_loss_start = random.uniform(0.01, 0.03)
        # P(loss | loss) - probability to continue losing (creates bursts)
        p_loss_continue = random.uniform(0.2, 0.4)

        in_loss_state = False

        for i in range(num_packets):
            # Markov state transition
            if in_loss_state:
                if random.random() >= p_loss_continue:
                    in_loss_state = False
            else:
                if random.random() < p_loss_start:
                    in_loss_state = True

            if in_loss_state:
                start = i * packet_size
                end = start + packet_size

                # Packet loss concealment strategies
                strategy = random.choice(['silence', 'interpolate', 'repeat'])

                if strategy == 'silence':
                    audio[start:end] = 0
                elif strategy == 'interpolate' and i > 0:
                    # Linear interpolation from previous packet
                    prev_end_val = audio[start-1] if start > 0 else 0
                    next_start_val = audio[end] if end < len(audio) else 0
                    audio[start:end] = np.linspace(prev_end_val, next_start_val, packet_size)
                elif strategy == 'repeat' and i > 0:
                    # Repeat previous packet (common concealment)
                    prev_start = (i - 1) * packet_size
                    audio[start:end] = audio[prev_start:start]

        return torch.tensor(np.ascontiguousarray(audio), dtype=torch.float32).unsqueeze(0)

    def _apply_jitter(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add timing jitter (sample-level timing variations).

        VoIP calls have variable packet arrival times, causing
        slight timing inconsistencies in the audio playback.
        """
        audio = waveform.numpy().squeeze()

        # Packet size (~20ms)
        packet_size = int(0.02 * self.config.SAMPLE_RATE)
        num_packets = len(audio) // packet_size

        # Maximum jitter in samples (0-50ms)
        max_jitter_ms = random.uniform(10, 50)
        max_jitter_samples = int(max_jitter_ms * self.config.SAMPLE_RATE / 1000)

        result = np.zeros_like(audio)
        write_pos = 0

        for i in range(num_packets):
            start = i * packet_size
            end = min(start + packet_size, len(audio))
            packet = audio[start:end]

            # Random delay for this packet
            jitter = random.randint(0, max_jitter_samples)

            # Calculate write position with jitter
            write_start = write_pos
            write_end = min(write_start + len(packet), len(result))

            # Copy packet to result (may overlap or have gaps)
            actual_len = write_end - write_start
            if actual_len > 0:
                result[write_start:write_end] = packet[:actual_len]

            # Advance write position (with some jitter variation)
            write_pos += len(packet) + random.randint(-jitter // 2, jitter // 2)
            write_pos = max(0, min(write_pos, len(result) - packet_size))

        return torch.tensor(np.ascontiguousarray(result), dtype=torch.float32).unsqueeze(0)

    def _simulate_vad_dtx(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Simulate Voice Activity Detection (VAD) and Discontinuous Transmission (DTX).

        Real phone codecs:
        - Clip speech onsets due to VAD detection delay (10-30ms)
        - Insert comfort noise during silence (instead of true silence)
        """
        audio = waveform.numpy().squeeze()

        # Frame size for VAD processing (~20ms)
        frame_size = int(0.02 * self.config.SAMPLE_RATE)
        num_frames = len(audio) // frame_size

        # Energy threshold for VAD
        energy_threshold = random.uniform(0.005, 0.02)

        # VAD delay in samples (10-30ms)
        vad_delay_ms = random.uniform(10, 30)
        vad_delay_samples = int(vad_delay_ms * self.config.SAMPLE_RATE / 1000)

        # Comfort noise level (very low)
        comfort_noise_level = random.uniform(0.001, 0.004)

        result = audio.copy()
        was_silent = True

        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio[start:end]

            # Calculate frame energy (RMS)
            energy = np.sqrt(np.mean(frame ** 2))

            is_silent = energy < energy_threshold

            if is_silent:
                # Replace silence with comfort noise
                comfort_noise = np.random.randn(frame_size) * comfort_noise_level
                result[start:end] = comfort_noise
                was_silent = True
            else:
                if was_silent:
                    # Transition from silence to speech - clip onset
                    clip_end = min(start + vad_delay_samples, end)
                    # Gradually fade in over the VAD delay period
                    fade_length = clip_end - start
                    if fade_length > 0:
                        fade_in = np.linspace(0, 1, fade_length)
                        result[start:clip_end] = result[start:clip_end] * fade_in
                was_silent = False

        return torch.tensor(np.ascontiguousarray(result), dtype=torch.float32).unsqueeze(0)

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
    import soundfile as sf

    if degrader is None:
        degrader = PhoneDegradation()

    # Load audio using soundfile (no FFmpeg required)
    audio_np, sample_rate = sf.read(input_path)

    # Convert to torch tensor with shape (channels, samples)
    # Use .copy() to avoid negative stride issues
    if audio_np.ndim == 1:
        waveform = torch.tensor(audio_np.copy(), dtype=torch.float32).unsqueeze(0)
    else:
        # Multi-channel: convert to (channels, samples)
        waveform = torch.tensor(audio_np.T.copy(), dtype=torch.float32)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Apply degradation
    degraded = degrader(waveform, sample_rate)

    # Save using soundfile
    sf.write(output_path, degraded.squeeze().numpy(), degrader.config.SAMPLE_RATE)


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
