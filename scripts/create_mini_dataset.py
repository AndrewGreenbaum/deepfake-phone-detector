#!/usr/bin/env python3
"""
Create a mini dataset for training the deepfake detector.

Downloads real voice samples from LJSpeech and generates
fake samples using Edge TTS (Microsoft's free TTS service).
"""

import asyncio
import os
import sys
import random
import urllib.request
import tarfile
from pathlib import Path

try:
    from .archive_safety import extract_tar
except ImportError:  # Direct script invocation
    from archive_safety import extract_tar
from tqdm import tqdm

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))


# Sample texts for TTS generation
SAMPLE_TEXTS = [
    "Hello, this is a test of the voice cloning detection system.",
    "The quick brown fox jumps over the lazy dog.",
    "Please verify your account by providing your social security number.",
    "This is an urgent message about your bank account security.",
    "I'm calling from technical support regarding your computer.",
    "Your package has been delayed and requires immediate attention.",
    "Congratulations, you've won a prize in our sweepstakes.",
    "This call is being recorded for quality assurance purposes.",
    "Please press one to speak with a representative.",
    "Your credit card has been compromised and needs verification.",
    "I'm following up on our conversation from yesterday.",
    "The weather forecast calls for rain later this evening.",
    "Would you like to hear about our special promotion today?",
    "Your appointment has been confirmed for tomorrow morning.",
    "Please hold while I transfer you to the next available agent.",
    "Thank you for calling, how may I assist you today?",
    "I apologize for the inconvenience this may have caused.",
    "Let me check on that information for you right away.",
    "Is there anything else I can help you with today?",
    "Have a great day and thank you for your business.",
]

# Edge TTS voices that sound natural
TTS_VOICES = [
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-AriaNeural",
    "en-US-DavisNeural",
    "en-US-AmberNeural",
    "en-US-AnaNeural",
    "en-US-AndrewNeural",
    "en-US-EmmaNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
]


def download_ljspeech_samples(output_dir: Path, num_samples: int = 200):
    """Download LJSpeech samples as real voice data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # LJSpeech direct download URL
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    tar_path = output_dir.parent / "ljspeech.tar.bz2"
    extract_dir = output_dir.parent / "LJSpeech-1.1"

    if not extract_dir.exists():
        print(f"Downloading LJSpeech dataset...")
        print(f"  URL: {url}")
        print(f"  This is ~2.6GB and may take a few minutes...")

        # Download with progress
        def report_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

        urllib.request.urlretrieve(url, tar_path, report_hook)
        print("\n  Download complete!")

        print("Extracting LJSpeech...")
        with tarfile.open(tar_path, "r:bz2") as tar:
            extract_tar(tar, output_dir.parent)
        print("  Extraction complete!")

        # Clean up tar file
        tar_path.unlink()

    # Copy samples to output directory
    wav_dir = extract_dir / "wavs"
    wav_files = list(wav_dir.glob("*.wav"))

    # Randomly select samples
    selected = random.sample(wav_files, min(num_samples, len(wav_files)))

    print(f"Copying {len(selected)} samples to {output_dir}...")
    for wav_file in tqdm(selected, desc="Copying real samples"):
        dest = output_dir / wav_file.name
        if not dest.exists():
            import shutil
            shutil.copy(wav_file, dest)

    return len(selected)


async def generate_tts_samples(output_dir: Path, num_samples: int = 200):
    """Generate fake voice samples using Edge TTS."""
    import edge_tts

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} TTS samples...")

    for i in tqdm(range(num_samples), desc="Generating fake samples"):
        # Random text and voice
        text = random.choice(SAMPLE_TEXTS)
        voice = random.choice(TTS_VOICES)

        output_file = output_dir / f"fake_{i:04d}_{voice.split('-')[1]}.wav"

        if output_file.exists():
            continue

        try:
            communicate = edge_tts.Communicate(text, voice)
            # Save as mp3 first (edge-tts default), then convert
            mp3_file = output_file.with_suffix(".mp3")
            await communicate.save(str(mp3_file))

            # Convert to wav using pydub
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(str(mp3_file))
            audio.export(str(output_file), format="wav")
            mp3_file.unlink()

        except Exception as e:
            print(f"\nError generating sample {i}: {e}")
            continue

    return len(list(output_dir.glob("*.wav")))


def apply_phone_degradation(input_dir: Path, output_dir: Path):
    """Apply phone degradation to all samples."""
    from phone_augment import PhoneDegradation, degrade_audio_file

    output_dir.mkdir(parents=True, exist_ok=True)

    degrader = PhoneDegradation()
    wav_files = list(input_dir.glob("*.wav"))

    print(f"Degrading {len(wav_files)} files to phone quality...")

    for wav_file in tqdm(wav_files, desc="Phone degradation"):
        output_file = output_dir / wav_file.name
        if output_file.exists():
            continue

        try:
            degrade_audio_file(str(wav_file), str(output_file), degrader)
        except Exception as e:
            print(f"\nError degrading {wav_file.name}: {e}")
            continue

    return len(list(output_dir.glob("*.wav")))


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create mini training dataset")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of samples per class")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--skip-real", action="store_true",
                        help="Skip downloading real samples")
    parser.add_argument("--skip-fake", action="store_true",
                        help="Skip generating fake samples")
    parser.add_argument("--skip-degrade", action="store_true",
                        help="Skip phone degradation")
    args = parser.parse_args()

    base_dir = Path(args.output)

    # Download real samples
    if not args.skip_real:
        real_count = download_ljspeech_samples(
            base_dir / "raw" / "real",
            args.num_samples
        )
        print(f"Real samples: {real_count}")

    # Generate fake samples
    if not args.skip_fake:
        fake_count = await generate_tts_samples(
            base_dir / "raw" / "fake",
            args.num_samples
        )
        print(f"Fake samples: {fake_count}")

    # Apply phone degradation
    if not args.skip_degrade:
        print("\nApplying phone degradation to real samples...")
        real_degraded = apply_phone_degradation(
            base_dir / "raw" / "real",
            base_dir / "phone" / "real"
        )

        print("\nApplying phone degradation to fake samples...")
        fake_degraded = apply_phone_degradation(
            base_dir / "raw" / "fake",
            base_dir / "phone" / "fake"
        )

        print(f"\nPhone-quality dataset ready:")
        print(f"  Real: {real_degraded} samples")
        print(f"  Fake: {fake_degraded} samples")

    print("\nDataset creation complete!")
    print(f"  Raw audio: {base_dir}/raw/")
    print(f"  Phone quality: {base_dir}/phone/")
    print("\nNext step: python model/train.py --data data/phone")


if __name__ == "__main__":
    asyncio.run(main())
