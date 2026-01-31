"""
Prepare LibriSpeech dataset for deepfake detection training.

LibriSpeech provides multi-speaker real audio (2400+ speakers in full dataset).
This is critical for training a generalizable detector.

Dataset variants:
- dev-clean: 40 speakers, ~5 hours (development)
- test-clean: 40 speakers, ~5 hours (testing)
- train-clean-100: 251 speakers, ~100 hours
- train-clean-360: 921 speakers, ~360 hours
- train-other-500: 1166 speakers, ~500 hours

For deepfake detection, we recommend:
- Start with dev-clean/test-clean for quick experiments
- Use train-clean-100 for good speaker diversity
"""

import os
import argparse
import tarfile
import random
from pathlib import Path
from typing import Optional
import shutil
from tqdm import tqdm

try:
    import requests
except ImportError:
    requests = None

# LibriSpeech download URLs
LIBRISPEECH_URL = "https://www.openslr.org/resources/12"

SUBSETS = {
    "dev-clean": {
        "url": f"{LIBRISPEECH_URL}/dev-clean.tar.gz",
        "speakers": 40,
        "hours": 5.4,
        "size_mb": 337,
    },
    "test-clean": {
        "url": f"{LIBRISPEECH_URL}/test-clean.tar.gz",
        "speakers": 40,
        "hours": 5.4,
        "size_mb": 346,
    },
    "train-clean-100": {
        "url": f"{LIBRISPEECH_URL}/train-clean-100.tar.gz",
        "speakers": 251,
        "hours": 100.6,
        "size_mb": 6300,
    },
    "train-clean-360": {
        "url": f"{LIBRISPEECH_URL}/train-clean-360.tar.gz",
        "speakers": 921,
        "hours": 363.6,
        "size_mb": 23000,
    },
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    if requests is None:
        print("Error: requests library not installed")
        print("Install with: pip install requests")
        return False

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except Exception as e:
        print(f"Download failed: {e}")
        return False


def extract_tarfile(tar_path: Path, output_dir: Path) -> bool:
    """Extract tar.gz file."""
    try:
        print(f"Extracting {tar_path.name}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
        return True
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def copy_samples(
    librispeech_dir: Path,
    output_dir: Path,
    subset: str,
    max_samples: Optional[int] = None,
    max_per_speaker: int = 20,
) -> int:
    """Copy audio samples from LibriSpeech to output directory.

    Args:
        librispeech_dir: Extracted LibriSpeech directory
        output_dir: Output directory for samples
        subset: Subset name (e.g., "dev-clean")
        max_samples: Maximum total samples to copy
        max_per_speaker: Maximum samples per speaker (for diversity)

    Returns:
        Number of samples copied
    """
    subset_dir = librispeech_dir / "LibriSpeech" / subset

    if not subset_dir.exists():
        print(f"Subset directory not found: {subset_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all FLAC files organized by speaker
    speaker_files = {}
    for speaker_dir in subset_dir.iterdir():
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            flac_files = list(speaker_dir.rglob("*.flac"))
            if flac_files:
                speaker_files[speaker_id] = flac_files

    print(f"Found {len(speaker_files)} speakers in {subset}")

    # Copy samples, limiting per speaker for diversity
    copied = 0
    total_available = sum(len(files) for files in speaker_files.values())

    with tqdm(total=min(max_samples or total_available, total_available)) as pbar:
        for speaker_id, files in speaker_files.items():
            # Shuffle and limit per speaker
            random.shuffle(files)
            speaker_samples = files[:max_per_speaker]

            for src_file in speaker_samples:
                if max_samples and copied >= max_samples:
                    break

                # Create unique filename with speaker ID
                dst_name = f"librispeech_{speaker_id}_{src_file.stem}.flac"
                dst_file = output_dir / dst_name

                shutil.copy2(src_file, dst_file)
                copied += 1
                pbar.update(1)

            if max_samples and copied >= max_samples:
                break

    return copied


def prepare_librispeech(
    output_dir: str = "data/raw/real",
    subset: str = "dev-clean",
    download_dir: str = "data/downloads",
    max_samples: int = 1000,
    max_per_speaker: int = 20,
    skip_download: bool = False,
):
    """Download and prepare LibriSpeech dataset.

    Args:
        output_dir: Where to put prepared samples
        subset: Which subset to download
        download_dir: Where to store downloaded archives
        max_samples: Maximum samples to extract
        max_per_speaker: Maximum samples per speaker
        skip_download: Skip download if archive exists
    """
    output_path = Path(output_dir)
    download_path = Path(download_dir)
    download_path.mkdir(parents=True, exist_ok=True)

    if subset not in SUBSETS:
        print(f"Unknown subset: {subset}")
        print(f"Available: {list(SUBSETS.keys())}")
        return

    info = SUBSETS[subset]
    archive_file = download_path / f"{subset}.tar.gz"
    extract_dir = download_path / "librispeech_extracted"

    print(f"LibriSpeech subset: {subset}")
    print(f"  Speakers: {info['speakers']}")
    print(f"  Hours: {info['hours']}")
    print(f"  Size: {info['size_mb']} MB")
    print()

    # Download if needed
    if not archive_file.exists() or not skip_download:
        if archive_file.exists() and skip_download:
            print(f"Archive exists: {archive_file}")
        else:
            print(f"Downloading {subset}...")
            if not download_file(info["url"], archive_file):
                return
    else:
        print(f"Using existing archive: {archive_file}")

    # Extract
    subset_extracted = extract_dir / "LibriSpeech" / subset
    if not subset_extracted.exists():
        if not extract_tarfile(archive_file, extract_dir):
            return
    else:
        print(f"Already extracted: {subset_extracted}")

    # Copy samples
    print(f"\nCopying samples to {output_path}...")
    print(f"Max samples: {max_samples}, Max per speaker: {max_per_speaker}")

    copied = copy_samples(
        extract_dir,
        output_path,
        subset,
        max_samples=max_samples,
        max_per_speaker=max_per_speaker,
    )

    print(f"\nDone! Copied {copied} samples to {output_path}")
    print("Next: Run degrade_to_phone.py to prepare for training")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LibriSpeech dataset for deepfake detection"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/real",
        help="Output directory for prepared samples",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="dev-clean",
        choices=list(SUBSETS.keys()),
        help="LibriSpeech subset to download",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="data/downloads",
        help="Directory for downloaded archives",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples to extract",
    )
    parser.add_argument(
        "--max-per-speaker",
        type=int,
        default=20,
        help="Maximum samples per speaker (for diversity)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if archive exists",
    )
    parser.add_argument(
        "--list-subsets",
        action="store_true",
        help="List available subsets and exit",
    )

    args = parser.parse_args()

    if args.list_subsets:
        print("Available LibriSpeech subsets:")
        print("-" * 60)
        for name, info in SUBSETS.items():
            print(f"  {name}:")
            print(f"    Speakers: {info['speakers']}")
            print(f"    Hours: {info['hours']}")
            print(f"    Size: {info['size_mb']} MB")
        return

    prepare_librispeech(
        output_dir=args.output,
        subset=args.subset,
        download_dir=args.download_dir,
        max_samples=args.max_samples,
        max_per_speaker=args.max_per_speaker,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
