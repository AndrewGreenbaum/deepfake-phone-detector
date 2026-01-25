"""
Test deepfake detection on audio files.
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model import get_model
from phone_augment import PhoneConfig, PhoneDegradation, PhoneMelSpectrogram


def load_model(checkpoint_path: str, model_name: str, device: str):
    """Load trained model from checkpoint."""
    model = get_model(model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_audio(audio_path: str, config: PhoneConfig, degrader: PhoneDegradation):
    """Load and preprocess audio file."""
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 8kHz if needed
    if sr != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, config.SAMPLE_RATE)
        waveform = resampler(waveform)

    # Apply phone degradation
    waveform_np = waveform.squeeze().numpy()
    degraded = degrader.degrade(waveform_np)

    return torch.from_numpy(degraded).float()


def detect_deepfake(
    audio_path: str,
    model: torch.nn.Module,
    mel_transform: PhoneMelSpectrogram,
    degrader: PhoneDegradation,
    config: PhoneConfig,
    device: str
) -> dict:
    """Run deepfake detection on a single audio file."""

    # Preprocess
    waveform = preprocess_audio(audio_path, config, degrader)

    # Convert to mel spectrogram
    mel_spec = mel_transform(waveform)

    # Ensure correct shape for model (batch, channel, mels, time)
    if mel_spec.dim() == 2:
        mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
    elif mel_spec.dim() == 3:
        mel_spec = mel_spec.unsqueeze(0)

    # Pad or truncate to expected time frames
    target_frames = 94
    if mel_spec.shape[-1] < target_frames:
        padding = target_frames - mel_spec.shape[-1]
        mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
    else:
        mel_spec = mel_spec[..., :target_frames]

    # Run inference
    mel_spec = mel_spec.to(device)
    with torch.no_grad():
        logits = model(mel_spec)
        probs = torch.softmax(logits, dim=1)

    real_prob = probs[0, 0].item()
    fake_prob = probs[0, 1].item()

    return {
        'file': audio_path,
        'prediction': 'FAKE' if fake_prob > real_prob else 'REAL',
        'confidence': max(real_prob, fake_prob),
        'real_probability': real_prob,
        'fake_probability': fake_prob
    }


def print_result(result: dict, verbose: bool = False):
    """Print detection result."""
    pred = result['prediction']
    conf = result['confidence']

    # Color output
    if pred == 'FAKE':
        color = '\033[91m'  # Red
        symbol = '⚠️ '
    else:
        color = '\033[92m'  # Green
        symbol = '✓ '
    reset = '\033[0m'

    print(f"{symbol}{color}{pred}{reset} ({conf:.1%} confidence) - {result['file']}")

    if verbose:
        print(f"   Real: {result['real_probability']:.3f}")
        print(f"   Fake: {result['fake_probability']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Test deepfake detection on audio files")
    parser.add_argument("audio", type=str, nargs='+', help="Audio file(s) to analyze")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="mobile",
                        choices=["phone_cnn", "temporal", "mobile", "attention"])
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed probabilities")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for detection")
    args = parser.parse_args()

    # Setup
    config = PhoneConfig()
    degrader = PhoneDegradation(config)
    mel_transform = PhoneMelSpectrogram(config)

    print(f"Loading model: {args.model}")
    model = load_model(args.checkpoint, args.model, args.device)
    print(f"Model loaded from {args.checkpoint}\n")

    # Process each file
    results = []
    for audio_path in args.audio:
        if not Path(audio_path).exists():
            print(f"File not found: {audio_path}")
            continue

        try:
            result = detect_deepfake(
                audio_path, model, mel_transform,
                degrader, config, args.device
            )
            results.append(result)
            print_result(result, args.verbose)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    # Summary
    if len(results) > 1:
        fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
        print(f"\n--- Summary ---")
        print(f"Total files: {len(results)}")
        print(f"Detected as FAKE: {fake_count}")
        print(f"Detected as REAL: {len(results) - fake_count}")


if __name__ == "__main__":
    main()
