"""
Generate fake audio samples using Bark TTS.
"""

import os
from pathlib import Path
import scipy.io.wavfile as wavfile

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


def generate_samples(output_dir: str, num_samples: int = 50):
    """Generate fake samples using Bark."""
    from bark import generate_audio, preload_models, SAMPLE_RATE
    from tqdm import tqdm

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading Bark models (this may take a few minutes on first run)...")
    preload_models()

    all_phrases = SCAM_PHRASES + NORMAL_PHRASES

    print(f"Generating {num_samples} fake audio samples...")
    for i in tqdm(range(num_samples)):
        text = all_phrases[i % len(all_phrases)]

        try:
            audio = generate_audio(text)
            output_file = output_path / f"bark_{i:04d}.wav"
            wavfile.write(str(output_file), SAMPLE_RATE, audio)
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue

    print(f"Generated samples saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/raw/fake/bark")
    parser.add_argument("--num", type=int, default=50)
    args = parser.parse_args()

    generate_samples(args.output, args.num)
