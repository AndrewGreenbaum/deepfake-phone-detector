"""
Extract and organize ASVspoof 2019 LA dataset.
"""

import os
import shutil
from pathlib import Path
import zipfile

DATA_DIR = Path(__file__).parent.parent / "data"


def extract_asvspoof(zip_path: str):
    """Extract ASVspoof LA dataset."""
    print(f"Extracting {zip_path}...")

    extract_dir = DATA_DIR / "downloads" / "LA"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / "downloads")

    print(f"Extracted to {extract_dir}")
    return extract_dir


def organize_dataset(la_dir: Path, max_samples: int = 5000):
    """
    Organize ASVspoof into real/fake directories.

    ASVspoof LA structure:
    - ASVspoof2019_LA_train/flac/ - training audio
    - ASVspoof2019_LA_dev/flac/ - dev audio
    - ASVspoof2019_LA_eval/flac/ - eval audio
    - ASVspoof2019_LA_cm_protocols/ - labels

    Labels file format: SPEAKER AUDIO_FILE - SYSTEM LABEL
    LABEL: bonafide (real) or spoof (fake)
    """

    real_dir = DATA_DIR / "raw" / "real" / "asvspoof"
    fake_dir = DATA_DIR / "raw" / "fake" / "asvspoof"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Find protocol files
    protocol_dir = la_dir / "ASVspoof2019_LA_cm_protocols"

    # Use training set for now
    train_protocol = protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    train_audio_dir = la_dir / "ASVspoof2019_LA_train" / "flac"

    if not train_protocol.exists():
        # Try alternative path
        for p in la_dir.rglob("*.txt"):
            if "train" in p.name.lower():
                train_protocol = p
                break

    if not train_audio_dir.exists():
        # Find audio directory
        for d in la_dir.rglob("flac"):
            if "train" in str(d).lower():
                train_audio_dir = d
                break

    print(f"Protocol: {train_protocol}")
    print(f"Audio dir: {train_audio_dir}")

    if not train_protocol.exists():
        print("ERROR: Protocol file not found!")
        print("Contents of LA dir:")
        for item in la_dir.rglob("*"):
            print(f"  {item}")
        return

    real_count = 0
    fake_count = 0

    with open(train_protocol, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            speaker = parts[0]
            audio_id = parts[1]
            label = parts[4]  # bonafide or spoof

            audio_file = train_audio_dir / f"{audio_id}.flac"

            if not audio_file.exists():
                continue

            if label == "bonafide" and real_count < max_samples:
                dest = real_dir / f"{audio_id}.flac"
                shutil.copy(audio_file, dest)
                real_count += 1
            elif label == "spoof" and fake_count < max_samples:
                dest = fake_dir / f"{audio_id}.flac"
                shutil.copy(audio_file, dest)
                fake_count += 1

            if real_count >= max_samples and fake_count >= max_samples:
                break

    print(f"Organized {real_count} real and {fake_count} fake samples")
    print(f"Real samples: {real_dir}")
    print(f"Fake samples: {fake_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, default="data/downloads/LA.zip")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples per class")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.is_absolute():
        zip_path = Path(__file__).parent.parent / args.zip

    # Extract
    la_dir = extract_asvspoof(str(zip_path))

    # Organize
    organize_dataset(la_dir, args.max_samples)

    print("\nDataset ready! Next steps:")
    print("1. Run: python scripts/degrade_to_phone.py")
    print("2. Run: python model/train.py")


if __name__ == "__main__":
    main()
