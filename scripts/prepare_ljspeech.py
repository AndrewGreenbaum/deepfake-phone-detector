"""Prepare LJSpeech dataset for training.

LJSpeech-1.1: Single female speaker, ~24 hours of studio-quality speech.
13,100 short audio clips from 7 non-fiction books.
Uses torchaudio's built-in LJSPEECH dataset loader for download,
then copies files directly to avoid FFmpeg dependencies.
"""
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def prepare_ljspeech(output_dir: str, max_samples: int = None):
    """Download and prepare LJSpeech samples.

    Args:
        output_dir: Path to output directory (e.g., data/raw/real)
        max_samples: Optional limit on number of samples to process
    """
    import torchaudio

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    download_dir = Path("data/downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LJSpeech dataset via torchaudio...")
    print("This will download ~2.6GB of audio files if not already cached.")

    # Use torchaudio to download LJSpeech
    torchaudio.datasets.LJSPEECH(
        root=str(download_dir),
        download=True
    )

    # Find the extracted wavs directory
    wavs_dir = download_dir / "LJSpeech-1.1" / "wavs"
    if not wavs_dir.exists():
        raise FileNotFoundError(f"Could not find LJSpeech wavs directory at {wavs_dir}")

    samples = list(wavs_dir.glob("*.wav"))
    total = len(samples)
    print(f"Found {total} LJSpeech samples")

    if max_samples:
        samples = samples[:max_samples]
        print(f"Limiting to {max_samples} samples")

    saved = 0
    for src in tqdm(samples, desc="Copying LJSpeech"):
        dst = output_path / src.name

        if not dst.exists():
            shutil.copy(src, dst)
            saved += 1

    print(f"Saved {saved} new LJSpeech samples to {output_path}")
    print(f"Total samples in output directory: {len(list(output_path.glob('*.wav')))}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LJSpeech dataset for training"
    )
    parser.add_argument(
        "--output",
        default="data/raw/real",
        help="Output directory for prepared samples"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all ~13,100)"
    )

    args = parser.parse_args()
    prepare_ljspeech(args.output, args.max_samples)


if __name__ == "__main__":
    main()
