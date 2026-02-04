"""
Generate fake audio samples using Microsoft Edge TTS.
Free, no API key required, 300+ voices.
"""

import asyncio
import os
from pathlib import Path
import edge_tts
from tqdm import tqdm
import random

# US English voices for variety
VOICES = [
    "en-US-AndrewNeural",
    "en-US-AriaNeural",
    "en-US-AvaNeural",
    "en-US-BrianNeural",
    "en-US-ChristopherNeural",
    "en-US-EmmaNeural",
    "en-US-EricNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-MichelleNeural",
    "en-US-RogerNeural",
    "en-US-SteffanNeural",
]

# Scam-like phrases
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
    "Hi, this is Microsoft support. We've detected a problem with your computer.",
    "Your Apple ID has been locked due to suspicious activity.",
    "This call is from the Social Security Administration regarding your benefits.",
    "We are calling about an outstanding warrant in your name.",
    "Your bank account will be suspended unless you verify your identity now.",
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
    "Hey, it's me. Give me a call when you get this.",
    "I'm running a few minutes late, be there soon.",
    "Just checking if you're free for lunch tomorrow.",
    "The project is coming along nicely, I'll update you later.",
    "Thanks for your help with that issue earlier.",
]


async def generate_sample(text: str, voice: str, output_path: str):
    """Generate a single audio sample."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


async def generate_samples(output_dir: str, num_samples: int = 100):
    """Generate fake samples using Edge TTS."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_phrases = SCAM_PHRASES + NORMAL_PHRASES

    print(f"Generating {num_samples} fake audio samples with Edge TTS...")

    for i in tqdm(range(num_samples)):
        text = all_phrases[i % len(all_phrases)]
        voice = random.choice(VOICES)
        output_file = output_path / f"edge_{i:04d}.mp3"

        try:
            await generate_sample(text, voice, str(output_file))
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue

    print(f"Generated {num_samples} samples saved to {output_dir}")
    print("Note: Files are MP3. Run degrade_to_phone.py to convert and degrade for training.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate fake samples using Edge TTS")
    parser.add_argument("--output", type=str, default="data/raw/fake/edge")
    parser.add_argument("--num", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(generate_samples(args.output, args.num))


if __name__ == "__main__":
    main()
