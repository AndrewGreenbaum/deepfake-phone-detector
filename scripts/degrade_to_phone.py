"""
Batch convert audio files to phone quality.
"""

import argparse
import os
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from phone_augment import PhoneDegradation, PhoneConfig, degrade_audio_file


def degrade_directory(input_dir: str, output_dir: str):
    """Degrade all audio files in a directory."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    degrader = PhoneDegradation()

    # Find all audio files
    audio_files = list(input_path.glob("**/*.wav")) + \
                  list(input_path.glob("**/*.flac")) + \
                  list(input_path.glob("**/*.mp3"))

    print(f"Found {len(audio_files)} audio files")

    for audio_file in tqdm(audio_files, desc="Degrading"):
        # Maintain directory structure
        rel_path = audio_file.relative_to(input_path)
        output_file = output_path / rel_path.with_suffix('.wav')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            degrade_audio_file(str(audio_file), str(output_file), degrader)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

    print(f"Degraded {len(audio_files)} files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert audio to phone quality")
    parser.add_argument("--input", type=str, default="../data/raw",
                        help="Input directory with clean audio")
    parser.add_argument("--output", type=str, default="../data/phone",
                        help="Output directory for phone-quality audio")
    args = parser.parse_args()

    degrade_directory(args.input, args.output)


if __name__ == "__main__":
    main()
