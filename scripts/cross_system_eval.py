"""
Cross-system evaluation for deepfake detection.

Evaluates a trained model on each TTS system separately to test generalization.
A model that truly learned deepfake features (not TTS-specific artifacts) should
perform well across all systems.

Usage:
    python scripts/cross_system_eval.py --checkpoint checkpoints/best_model.pt --data data/phone

Expected directory structure:
    data/phone/
        real/           # Real audio samples
        fake/
            edge/       # Edge TTS samples
            bark/       # Bark samples
            elevenlabs/ # ElevenLabs samples
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from dataset import PhoneAudioDataset
from model import get_model
from phone_augment import PhoneConfig


def discover_tts_systems(data_path: str) -> list:
    """Discover available TTS systems in the data directory."""
    fake_dir = Path(data_path) / "fake"
    if not fake_dir.exists():
        return []

    systems = []
    for item in fake_dir.iterdir():
        if item.is_dir():
            # Check if directory has audio files
            audio_files = list(item.glob("*.wav")) + list(item.glob("*.mp3")) + list(item.glob("*.flac"))
            if audio_files:
                systems.append(item.name)

    return sorted(systems)


def count_samples_by_system(data_path: str) -> dict:
    """Count samples for each TTS system."""
    counts = {}
    fake_dir = Path(data_path) / "fake"

    if not fake_dir.exists():
        return counts

    for item in fake_dir.iterdir():
        if item.is_dir():
            audio_files = list(item.rglob("*.wav")) + list(item.rglob("*.mp3")) + list(item.rglob("*.flac"))
            if audio_files:
                counts[item.name] = len(audio_files)

    # Count real samples
    real_dir = Path(data_path) / "real"
    if real_dir.exists():
        real_files = list(real_dir.rglob("*.wav")) + list(real_dir.rglob("*.mp3")) + list(real_dir.rglob("*.flac"))
        counts["real"] = len(real_files)

    return counts


def evaluate_on_system(
    model: torch.nn.Module,
    data_path: str,
    tts_system: str,
    device: torch.device,
    batch_size: int = 32,
    apply_degradation: bool = False,
) -> dict:
    """Evaluate model on a specific TTS system.

    Args:
        model: Trained model
        data_path: Data directory
        tts_system: TTS system to evaluate on
        device: Device to run on
        batch_size: Batch size
        apply_degradation: Whether to apply phone degradation

    Returns:
        Dictionary with evaluation metrics
    """
    config = PhoneConfig()

    # Create dataset for specific TTS system
    dataset = PhoneAudioDataset(
        data_path,
        config=config,
        apply_degradation=apply_degradation,
        augment=False,
        holdout_tts=tts_system,
        holdout_only=True,
    )

    if len(dataset) == 0:
        return {
            "system": tts_system,
            "total": 0,
            "accuracy": 0,
            "fake_correct": 0,
            "fake_total": 0,
            "detection_rate": 0,
        }

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model.eval()
    correct, total = 0, 0
    fake_correct, fake_total = 0, 0
    predictions = []
    labels_list = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Track fake detection
            fake_mask = labels == 1
            if fake_mask.any():
                fake_total += fake_mask.sum().item()
                fake_correct += (predicted[fake_mask] == 1).sum().item()

            predictions.extend(predicted.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

    accuracy = 100. * correct / total if total > 0 else 0
    detection_rate = 100. * fake_correct / fake_total if fake_total > 0 else 0

    return {
        "system": tts_system,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "fake_correct": fake_correct,
        "fake_total": fake_total,
        "detection_rate": detection_rate,
    }


def evaluate_on_real(
    model: torch.nn.Module,
    data_path: str,
    device: torch.device,
    batch_size: int = 32,
    apply_degradation: bool = False,
    max_samples: int = 500,
) -> dict:
    """Evaluate model on real samples (false positive rate).

    Args:
        model: Trained model
        data_path: Data directory
        device: Device to run on
        batch_size: Batch size
        apply_degradation: Whether to apply phone degradation
        max_samples: Maximum real samples to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    config = PhoneConfig()
    real_dir = Path(data_path) / "real"

    if not real_dir.exists():
        return {"system": "real", "total": 0, "accuracy": 0, "false_positive_rate": 0}

    # Load real samples manually (since holdout doesn't apply to real)
    from dataset import PhoneMelSpectrogram, PhoneDegradation
    import soundfile as sf
    import numpy as np

    mel_transform = PhoneMelSpectrogram(config)
    degrader = PhoneDegradation(config) if apply_degradation else None

    # Get audio files
    audio_files = list(real_dir.rglob("*.wav")) + list(real_dir.rglob("*.flac")) + list(real_dir.rglob("*.mp3"))
    audio_files = audio_files[:max_samples]

    if not audio_files:
        return {"system": "real", "total": 0, "accuracy": 0, "false_positive_rate": 0}

    model.eval()
    correct, total = 0, 0
    false_positives = 0

    with torch.no_grad():
        for filepath in tqdm(audio_files, desc="Evaluating real samples"):
            try:
                audio, sr = sf.read(str(filepath))
                audio = np.ascontiguousarray(audio.astype(np.float32))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                waveform = torch.from_numpy(audio).unsqueeze(0)

                if degrader:
                    waveform = degrader(waveform, sr)
                elif sr != config.SAMPLE_RATE:
                    import torchaudio
                    waveform = torchaudio.functional.resample(waveform, sr, config.SAMPLE_RATE)

                # Pad/truncate
                target = config.NUM_SAMPLES
                if waveform.shape[1] < target:
                    waveform = torch.nn.functional.pad(waveform, (0, target - waveform.shape[1]))
                else:
                    waveform = waveform[:, :target]

                mel_spec = mel_transform(waveform)
                mel_spec = mel_spec.unsqueeze(0).to(device)

                outputs = model(mel_spec)
                _, predicted = outputs.max(1)

                total += 1
                if predicted.item() == 0:  # Correctly identified as real
                    correct += 1
                else:
                    false_positives += 1

            except Exception as e:
                continue

    accuracy = 100. * correct / total if total > 0 else 0
    fpr = 100. * false_positives / total if total > 0 else 0

    return {
        "system": "real",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "false_positive_rate": fpr,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Cross-system evaluation for deepfake detection"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/phone",
        help="Data directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="attention",
        choices=["phone_cnn", "temporal", "mobile", "attention"],
        help="Model architecture",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--degrade",
        action="store_true",
        help="Apply phone degradation (use if data is not pre-degraded)",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover TTS systems
    print(f"\nScanning {args.data} for TTS systems...")
    systems = discover_tts_systems(args.data)
    counts = count_samples_by_system(args.data)

    if not systems:
        print("No TTS systems found!")
        print("Expected structure: data/fake/edge/, data/fake/bark/, etc.")
        return

    print(f"Found {len(systems)} TTS systems:")
    for system in systems:
        print(f"  - {system}: {counts.get(system, 0)} samples")
    print(f"  - real: {counts.get('real', 0)} samples")

    # Load model
    print(f"\nLoading model: {args.model}")
    model = get_model(args.model)

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint (val acc: {checkpoint.get('val_acc', 'N/A')})")

    # Evaluate on each system
    print(f"\n{'='*70}")
    print("Cross-System Evaluation Results")
    print(f"{'='*70}")

    results = []

    # Evaluate on real samples first
    print("\nEvaluating on real samples...")
    real_result = evaluate_on_real(
        model, args.data, device,
        batch_size=args.batch_size,
        apply_degradation=args.degrade,
    )
    results.append(real_result)

    # Evaluate on each TTS system
    for system in systems:
        print(f"\nEvaluating on {system}...")
        result = evaluate_on_system(
            model, args.data, system, device,
            batch_size=args.batch_size,
            apply_degradation=args.degrade,
        )
        results.append(result)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'System':<15} {'Samples':<10} {'Accuracy':<12} {'Detection':<12} {'Status'}")
    print(f"{'='*70}")

    total_correct = 0
    total_samples = 0
    all_pass = True

    for r in results:
        system = r["system"]
        total = r["total"]

        if system == "real":
            # For real samples, show false positive rate
            acc = r["accuracy"]
            fpr = r.get("false_positive_rate", 0)
            status = "PASS" if fpr < 15 else "FAIL"
            print(f"{system:<15} {total:<10} {acc:.1f}%{'':<6} FPR: {fpr:.1f}%{'':<4} {status}")
            if fpr >= 15:
                all_pass = False
        else:
            acc = r["accuracy"]
            det = r["detection_rate"]
            status = "PASS" if det >= 85 else "WARN" if det >= 70 else "FAIL"
            print(f"{system:<15} {total:<10} {acc:.1f}%{'':<6} {det:.1f}%{'':<6} {status}")
            if det < 85:
                all_pass = False

        total_correct += r.get("correct", 0)
        total_samples += total

    print(f"{'='*70}")

    # Overall metrics
    overall_acc = 100. * total_correct / total_samples if total_samples > 0 else 0
    print(f"\nOverall Accuracy: {overall_acc:.1f}%")

    # Summary
    print(f"\n{'='*70}")
    if all_pass:
        print("SUCCESS: Model generalizes across all TTS systems!")
        print("The Bitter Lesson validated: diverse data > clever features")
    else:
        print("WARNING: Model may be overfitting to specific TTS artifacts")
        print("\nRecommendations:")
        print("  1. Add more diverse TTS samples to training data")
        print("  2. Use --holdout-tts flag during training to test generalization")
        print("  3. Consider data augmentation or regularization")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
