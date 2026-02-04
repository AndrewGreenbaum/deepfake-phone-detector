"""
Generate fake audio samples using Google TTS (gTTS).
Free, no API key required, adds TTS diversity.
"""

import os
from pathlib import Path
from gtts import gTTS
from tqdm import tqdm
import random

# Phrases for variety (different from Edge TTS)
PHRASES = [
    # Phone scam variations
    "Hello, this is a representative from your credit card company.",
    "We have detected unauthorized access to your bank account.",
    "Please verify your account information to avoid suspension.",
    "This is an automated message about your recent purchase.",
    "Your account security has been compromised, please take action.",
    "We are calling regarding a refund you are entitled to receive.",
    "There has been suspicious activity detected on your account.",
    "Press one now to speak with a fraud prevention specialist.",
    "Your payment method needs to be updated immediately.",
    "This is the last warning about your overdue balance.",

    # Business calls
    "Good morning, I am calling from the sales department.",
    "Thank you for your interest in our services.",
    "I wanted to follow up regarding your recent inquiry.",
    "Our records show that you requested more information.",
    "I have some important updates about your account.",
    "Your appointment has been confirmed for tomorrow.",
    "Please call us back at your earliest convenience.",
    "We have a special offer available for you today.",
    "I will send the details to your email address.",
    "Let me transfer you to the appropriate department.",

    # Casual conversation
    "Hey, just wanted to check in and see how you are doing.",
    "I will be running a few minutes late to our meeting.",
    "Can you give me a call back when you get a chance?",
    "I hope everything is going well on your end.",
    "Let me know what time works best for you.",
    "I have some news I wanted to share with you.",
    "Thanks for getting back to me so quickly.",
    "I was thinking about what we discussed earlier.",
    "Do you have any updates on that project?",
    "I will send you the information right away.",

    # Technical support
    "Hello, I am calling from technical support.",
    "Your computer appears to have a security issue.",
    "I can help you resolve this problem today.",
    "Please allow me to access your computer remotely.",
    "We have detected malware on your system.",
    "Your software license is about to expire.",
    "I will guide you through the troubleshooting steps.",
    "Please do not restart your computer during this process.",
    "We need to verify your identity for security purposes.",
    "The issue has been resolved successfully.",

    # General announcements
    "Your order has been shipped and is on its way.",
    "The delivery is scheduled for tomorrow afternoon.",
    "Please have your identification ready for verification.",
    "Your subscription will renew automatically next month.",
    "We appreciate your continued business with us.",
    "Your feedback is very important to us.",
    "Please complete the survey at the end of this call.",
    "Your account has been successfully updated.",
    "Thank you for choosing our services.",
    "We look forward to serving you again soon.",
]

# Languages/accents for variety
ACCENTS = [
    ('en', 'com'),      # US English
    ('en', 'co.uk'),    # British English
    ('en', 'com.au'),   # Australian English
    ('en', 'co.in'),    # Indian English
]


def generate_samples(output_dir: str, num_samples: int = 100):
    """Generate fake samples using Google TTS."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} fake audio samples with Google TTS...")

    for i in tqdm(range(num_samples)):
        text = PHRASES[i % len(PHRASES)]
        lang, tld = random.choice(ACCENTS)
        output_file = output_path / f"gtts_{i:04d}.mp3"

        try:
            tts = gTTS(text=text, lang=lang, tld=tld)
            tts.save(str(output_file))
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue

    generated = len(list(output_path.glob("*.mp3")))
    print(f"Generated {generated} samples saved to {output_dir}")
    print("Note: Files are MP3. Run degrade_to_phone.py to convert and degrade for training.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate fake samples using Google TTS")
    parser.add_argument("--output", type=str, default="data/raw/fake/gtts")
    parser.add_argument("--num", type=int, default=100)
    args = parser.parse_args()

    generate_samples(args.output, args.num)


if __name__ == "__main__":
    main()
