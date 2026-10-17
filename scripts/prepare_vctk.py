"""Prepare VCTK dataset for training.

VCTK Corpus: 110 speakers with various accents, studio-quality recordings.
Uses torchaudio's built-in VCTK_092 dataset loader.
"""
import argparse
import shutil
from pathlib import Path

try:
    from .archive_safety import extract_zip
except ImportError:  # Direct script invocation
    from archive_safety import extract_zip
from tqdm import tqdm
import soundfile as sf


def prepare_vctk_torchaudio(output_dir: str, max_samples: int = None):
    """Download and prepare VCTK samples using torchaudio.

    Args:
        output_dir: Path to output directory (e.g., data/raw/real/vctk)
        max_samples: Optional limit on number of samples to process
    """
    import torchaudio

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    download_dir = Path("data/downloads")
    download_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VCTK dataset via torchaudio...")
    print("This will download ~10GB of audio files if not already cached.")

    # Use torchaudio's VCTK_092 dataset - it handles download automatically
    dataset = torchaudio.datasets.VCTK_092(
        root=str(download_dir),
        download=True,
        mic_id="mic1"  # Use mic1 for cleaner audio
    )

    total = len(dataset)
    print(f"Found {total} VCTK samples")

    indices = range(total)
    if max_samples:
        indices = range(min(max_samples, total))
        print(f"Limiting to {max_samples} samples")

    saved = 0
    for i in tqdm(indices, desc="Processing VCTK"):
        try:
            waveform, sample_rate, utterance, speaker_id, utterance_id = dataset[i]

            # Create unique filename: vctk_<speaker>_<utterance_id>.wav
            dst = output_path / f"vctk_{speaker_id}_{utterance_id}.wav"

            if not dst.exists():
                # Save as WAV file
                torchaudio.save(str(dst), waveform, sample_rate)
                saved += 1
        except Exception as e:
            print(f"Warning: Failed to process sample {i}: {e}")
            continue

    print(f"Saved {saved} new VCTK samples to {output_path}")
    print(f"Total samples in output directory: {len(list(output_path.glob('*.wav')))}")


def prepare_vctk_from_zip(zip_path: str, output_dir: str, max_samples: int = None):
    """Extract and prepare VCTK samples from downloaded ZIP file.

    Args:
        zip_path: Path to VCTK-Corpus-0.92.zip
        output_dir: Path to output directory (e.g., data/raw/real/vctk)
        max_samples: Optional limit on number of samples to copy
    """
    import zipfile

    zip_path = Path(zip_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    print(f"Extracting VCTK from {zip_path}...")
    extract_dir = zip_path.parent / "vctk_extracted"

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List files matching our pattern
        members = [m for m in zf.namelist() if '_mic1.flac' in m]
        print(f"Found {len(members)} mic1 audio files in archive")

        if max_samples:
            members = members[:max_samples]
            print(f"Limiting to {max_samples} samples")

        extract_zip(zf, extract_dir, members)
        for member in tqdm(members, desc="Preparing VCTK"):

            # Get the extracted file path
            src = extract_dir / member

            # Parse speaker from path like VCTK-Corpus-0.92/wav48_silence_trimmed/p225/p225_001_mic1.flac
            parts = Path(member).parts
            speaker = parts[-2] if len(parts) >= 2 else "unknown"
            stem = Path(member).stem

            dst = output_path / f"vctk_{speaker}_{stem}.flac"

            if not dst.exists():
                shutil.move(str(src), str(dst))

    print(f"Total samples in output directory: {len(list(output_path.glob('*.flac')))}")


def prepare_vctk_local(input_dir: str, output_dir: str, max_samples: int = None):
    """Copy VCTK samples from local extraction to training directory.

    Args:
        input_dir: Path to extracted VCTK-Corpus-0.92 directory
        output_dir: Path to output directory (e.g., data/raw/real/vctk)
        max_samples: Optional limit on number of samples to copy
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # VCTK has structure: wav48_silence_trimmed/p225/p225_001_mic1.flac
    # Use mic1 recordings (closer microphone, cleaner audio)
    wav_dir = input_path / "wav48_silence_trimmed"

    if not wav_dir.exists():
        # Try alternative structure
        wav_dir = input_path / "wav48"
        if not wav_dir.exists():
            raise FileNotFoundError(
                f"Could not find wav48_silence_trimmed or wav48 in {input_path}. "
                "Make sure the VCTK corpus is properly extracted."
            )

    # Find all mic1 recordings (or all files if using wav48)
    if (input_path / "wav48_silence_trimmed").exists():
        samples = list(wav_dir.glob("*/p*_mic1.flac"))
    else:
        samples = list(wav_dir.glob("*/*.wav"))

    if not samples:
        raise FileNotFoundError(f"No audio files found in {wav_dir}")

    print(f"Found {len(samples)} VCTK samples")

    if max_samples:
        samples = samples[:max_samples]
        print(f"Limiting to {max_samples} samples")

    copied = 0
    for src in tqdm(samples, desc="Copying VCTK"):
        speaker = src.parent.name
        # Create unique filename: vctk_<speaker>_<original_stem>.<ext>
        dst = output_path / f"vctk_{speaker}_{src.stem}{src.suffix}"

        if not dst.exists():
            shutil.copy(src, dst)
            copied += 1

    print(f"Copied {copied} new VCTK samples to {output_path}")
    print(f"Total samples in output directory: {len(list(output_path.iterdir()))}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VCTK dataset for training"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to extracted VCTK corpus (if using --local)"
    )
    parser.add_argument(
        "--zip",
        default=None,
        help="Path to VCTK-Corpus-0.92.zip file"
    )
    parser.add_argument(
        "--output",
        default="data/raw/real/vctk",
        help="Output directory for prepared samples"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use locally extracted VCTK corpus"
    )

    args = parser.parse_args()

    if args.local:
        if not args.input:
            args.input = "data/downloads/vctk_extracted/VCTK-Corpus-0.92"
        prepare_vctk_local(args.input, args.output, args.max_samples)
    elif args.zip:
        prepare_vctk_from_zip(args.zip, args.output, args.max_samples)
    else:
        # Default: use torchaudio which has good download infrastructure
        prepare_vctk_torchaudio(args.output, args.max_samples)


if __name__ == "__main__":
    main()
