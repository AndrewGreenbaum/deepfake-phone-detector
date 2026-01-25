"""
Download and prepare datasets for phone deepfake detection.
"""

import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def setup_directories():
    """Create data directories."""
    dirs = [
        DATA_DIR / "raw" / "real",
        DATA_DIR / "raw" / "fake",
        DATA_DIR / "phone" / "real",
        DATA_DIR / "phone" / "fake",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Created directories in {DATA_DIR}")


def print_instructions():
    """Print dataset collection instructions."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              Phone Deepfake Detector - Dataset Setup              ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                   ║
    ║  STEP 1: Collect REAL voice samples                              ║
    ║  ─────────────────────────────────────────────────────────────   ║
    ║  Options:                                                         ║
    ║  • ASVspoof 2019/2021 (bonafide samples)                         ║
    ║    https://www.asvspoof.org/                                     ║
    ║  • LibriSpeech (clean speech)                                    ║
    ║    https://www.openslr.org/12                                    ║
    ║  • VoxCeleb (celebrity voices)                                   ║
    ║    https://www.robots.ox.ac.uk/~vgg/data/voxceleb/               ║
    ║  • Record your own voice samples                                  ║
    ║                                                                   ║
    ║  Save to: data/raw/real/                                         ║
    ║                                                                   ║
    ║  STEP 2: Collect FAKE (AI-generated) samples                     ║
    ║  ─────────────────────────────────────────────────────────────   ║
    ║  Options:                                                         ║
    ║  • ASVspoof 2019/2021 (spoof samples)                            ║
    ║  • Generate with ElevenLabs (https://elevenlabs.io)              ║
    ║  • Generate with Bark (free, local):                             ║
    ║    pip install git+https://github.com/suno-ai/bark.git           ║
    ║  • Generate with Coqui TTS:                                      ║
    ║    pip install TTS                                               ║
    ║  • Generate with OpenAI TTS API                                  ║
    ║                                                                   ║
    ║  Save to: data/raw/fake/                                         ║
    ║                                                                   ║
    ║  STEP 3: Apply phone degradation                                 ║
    ║  ─────────────────────────────────────────────────────────────   ║
    ║  Run: python scripts/degrade_to_phone.py                         ║
    ║                                                                   ║
    ║  This simulates phone call quality (8kHz, GSM codec, noise)      ║
    ║                                                                   ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


def generate_fake_with_bark(num_samples: int = 50):
    """Generate fake samples using Bark TTS."""
    try:
        from bark import generate_audio, preload_models, SAMPLE_RATE
        import scipy.io.wavfile as wavfile
        from tqdm import tqdm
    except ImportError:
        print("Install Bark: pip install git+https://github.com/suno-ai/bark.git")
        return

    print(f"Generating {num_samples} fake samples with Bark...")
    preload_models()

    output_dir = DATA_DIR / "raw" / "fake" / "bark"
    output_dir.mkdir(parents=True, exist_ok=True)

    texts = [
        "Hello, this is your bank calling about suspicious activity.",
        "We need to verify your account information immediately.",
        "Press one to speak with a representative.",
        "Your package could not be delivered, please confirm your address.",
        "This is an urgent message about your warranty.",
    ]

    for i in tqdm(range(num_samples)):
        text = texts[i % len(texts)]
        audio = generate_audio(text)
        wavfile.write(output_dir / f"bark_{i:04d}.wav", SAMPLE_RATE, audio)

    print(f"Generated {num_samples} samples in {output_dir}")


def main():
    setup_directories()
    print_instructions()

    response = input("\nGenerate fake samples with Bark? [y/N]: ")
    if response.lower() == 'y':
        num = int(input("How many samples? [50]: ") or "50")
        generate_fake_with_bark(num)

    print("\nNext steps:")
    print("1. Add real voice samples to data/raw/real/")
    print("2. Add fake voice samples to data/raw/fake/")
    print("3. Run: python scripts/degrade_to_phone.py")
    print("4. Run: python model/train.py")


if __name__ == "__main__":
    main()
