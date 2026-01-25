#!/usr/bin/env python3
"""
Create full-scale dataset for training.

The Bitter Lesson: Scale > Tricks

This script uses ALL available data:
- All 13K LJSpeech samples as real voices
- Generates additional fake samples via TTS
- Applies phone degradation to everything
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import numpy as np

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from phone_augment import PhoneConfig, PhoneDegradation


def get_ljspeech_samples(ljspeech_dir: Path, max_samples: int = None):
    """Get all LJSpeech samples."""
    wavs_dir = ljspeech_dir / "wavs"
    samples = list(wavs_dir.glob("*.wav"))

    if max_samples:
        random.shuffle(samples)
        samples = samples[:max_samples]

    return samples


def apply_phone_degradation_batch(
    input_files: list,
    output_dir: Path,
    config: PhoneConfig = None
):
    """Apply phone degradation to a batch of files."""
    import torch

    if config is None:
        config = PhoneConfig()

    degrader = PhoneDegradation(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for filepath in tqdm(input_files, desc="Applying phone degradation"):
        try:
            # Load audio
            audio, sr = sf.read(filepath)

            # Convert to float32 and ensure contiguous
            audio = np.ascontiguousarray(audio.astype(np.float32))

            # Handle stereo -> mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Convert to torch tensor with channel dimension (required by PhoneDegradation)
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

            # Apply degradation
            degraded = degrader(waveform, sr)

            # Convert to numpy
            degraded = degraded.numpy().squeeze()

            # Save
            output_path = output_dir / f"{filepath.stem}.wav"
            sf.write(str(output_path), degraded, config.SAMPLE_RATE)
            successful += 1

        except Exception as e:
            failed += 1
            if failed < 5:  # Only print first few errors
                print(f"Error processing {filepath}: {e}")

    return successful, failed


def main():
    parser = argparse.ArgumentParser(description="Create full-scale dataset")
    parser.add_argument("--max-real", type=int, default=5000,
                        help="Max real samples to use (default: 5000)")
    parser.add_argument("--output-dir", type=str, default="data/phone_full",
                        help="Output directory for phone-degraded data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Paths
    base_dir = Path(__file__).parent.parent
    ljspeech_dir = base_dir / "data" / "raw" / "LJSpeech-1.1"
    fake_dir = base_dir / "data" / "raw" / "fake"
    output_dir = base_dir / args.output_dir

    print("=" * 60)
    print("Creating Full-Scale Phone Dataset")
    print("The Bitter Lesson: More data beats clever tricks")
    print("=" * 60)

    # Get LJSpeech samples (real)
    print(f"\nStep 1: Loading LJSpeech samples...")
    real_samples = get_ljspeech_samples(ljspeech_dir, args.max_real)
    print(f"  Found {len(real_samples)} real samples")

    # Get fake samples
    print(f"\nStep 2: Loading fake samples...")
    fake_samples = list(fake_dir.glob("**/*.wav")) + list(fake_dir.glob("**/*.mp3"))
    print(f"  Found {len(fake_samples)} fake samples")

    # Apply phone degradation
    print(f"\nStep 3: Applying phone degradation to real samples...")
    real_output = output_dir / "real"
    real_success, real_fail = apply_phone_degradation_batch(real_samples, real_output)
    print(f"  Processed: {real_success} success, {real_fail} failed")

    print(f"\nStep 4: Applying phone degradation to fake samples...")
    fake_output = output_dir / "fake"
    fake_success, fake_fail = apply_phone_degradation_batch(fake_samples, fake_output)
    print(f"  Processed: {fake_success} success, {fake_fail} failed")

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Real samples: {real_success}")
    print(f"Fake samples: {fake_success}")
    print(f"Total: {real_success + fake_success}")
    print(f"Output: {output_dir}")

    # Class balance info
    if real_success > 0 and fake_success > 0:
        ratio = real_success / fake_success
        print(f"\nClass ratio (real:fake): {ratio:.2f}:1")
        if ratio > 2:
            print("WARNING: Significant class imbalance!")
            print("Consider: --no-class-weights flag OR generate more fakes")

    print("\nNext steps:")
    print(f"  python model/train.py --data {args.output_dir} --model attention --epochs 100")


if __name__ == "__main__":
    main()
