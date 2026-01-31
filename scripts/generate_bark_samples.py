"""
Generate fake audio samples using Bark TTS.

Bark is a transformer-based text-to-audio model that can generate
realistic speech with various speaker presets.

Install: pip install git+https://github.com/suno-ai/bark.git
Note: Requires ~12GB VRAM for full model, ~5GB for small model
"""

import os
from pathlib import Path
import numpy as np
import soundfile as sf

# Scam-like phrases for training
SCAM_PHRASES = [
    "Hello, this is your bank calling about suspicious activity on your account.",
    "We need to verify your social security number immediately.",
    "Press one to speak with a representative about your account.",
    "Your package could not be delivered, please confirm your address.",
    "This is an urgent message about your car's extended warranty.",
    "You have won a free cruise vacation, press one to claim your prize.",
    "There is a problem with your tax return, call us back immediately.",
    "Your computer has been infected with a virus, do not turn it off.",
    "This is the IRS calling about unpaid taxes on your account.",
    "Your Amazon order has been cancelled due to payment issues.",
    "Hello, I am calling from tech support about your internet connection.",
    "We detected unusual login attempts on your email account.",
    "Your credit card has been compromised, please verify your information.",
    "This is a final notice about your student loan forgiveness program.",
    "You have been selected for a special government grant program.",
]

# Normal conversation phrases
NORMAL_PHRASES = [
    "Hi, how are you doing today?",
    "I was just calling to check in and see how things are going.",
    "Did you get my message from earlier?",
    "Let me know when you have a chance to talk.",
    "I wanted to follow up on our conversation from yesterday.",
    "Thanks for getting back to me so quickly.",
    "I hope you're having a great day.",
    "Just wanted to say hello and catch up.",
    "Can we schedule a meeting for next week?",
    "I'll send you the details in an email.",
]

# Bark speaker presets for variety
# See: https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c
SPEAKER_PRESETS = [
    "v2/en_speaker_0",
    "v2/en_speaker_1",
    "v2/en_speaker_2",
    "v2/en_speaker_3",
    "v2/en_speaker_4",
    "v2/en_speaker_5",
    "v2/en_speaker_6",
    "v2/en_speaker_7",
    "v2/en_speaker_8",
    "v2/en_speaker_9",
]


def generate_samples(
    output_dir: str,
    num_samples: int = 50,
    use_small_model: bool = False,
):
    """Generate fake samples using Bark.

    Args:
        output_dir: Directory to save generated audio
        num_samples: Number of samples to generate
        use_small_model: Use small model (less VRAM, lower quality)
    """
    import random
    from tqdm import tqdm

    # Configure Bark before importing
    if use_small_model:
        os.environ["SUNO_USE_SMALL_MODELS"] = "True"

    try:
        from bark import generate_audio, preload_models, SAMPLE_RATE
    except ImportError:
        print("Bark not installed. Install with:")
        print("  pip install git+https://github.com/suno-ai/bark.git")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading Bark models (this may take a few minutes on first run)...")
    print("Note: Requires ~12GB VRAM for full model, ~5GB for small model")
    preload_models()

    all_phrases = SCAM_PHRASES + NORMAL_PHRASES

    print(f"Generating {num_samples} fake audio samples with Bark...")
    print(f"Output: {output_dir}")

    generated = 0

    for i in tqdm(range(num_samples)):
        text = all_phrases[i % len(all_phrases)]
        speaker = random.choice(SPEAKER_PRESETS)

        try:
            # Generate audio with speaker preset for variety
            audio = generate_audio(
                text,
                history_prompt=speaker,
            )

            # Validate audio is not empty or silent
            if audio is None or len(audio) == 0:
                print(f"Warning: Empty audio for sample {i}, skipping")
                continue

            # Check for near-silence (RMS < threshold)
            rms = np.sqrt(np.mean(audio**2))
            if rms < 0.001:
                print(f"Warning: Near-silent audio for sample {i}, skipping")
                continue

            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95

            output_file = output_path / f"bark_{i:04d}.wav"

            # Use soundfile for proper float32 handling
            sf.write(str(output_file), audio, SAMPLE_RATE)
            generated += 1

        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue

    print(f"\nGenerated {generated}/{num_samples} samples saved to {output_dir}")
    print("Note: Run degrade_to_phone.py to convert for training.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate fake samples using Bark TTS"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/fake/bark",
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use small model (less VRAM required, lower quality)",
    )

    args = parser.parse_args()

    generate_samples(args.output, args.num, use_small_model=args.small)
