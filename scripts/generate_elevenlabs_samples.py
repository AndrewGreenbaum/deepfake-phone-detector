"""
Generate fake audio samples using ElevenLabs TTS API.
Requires API key set as ELEVENLABS_API_KEY environment variable.

Features:
- Voice settings variation for diverse synthesis styles
- Multi-model generation across all ElevenLabs models
- Dynamic voice discovery from API
- Metadata tracking for generated samples
- Dry-run cost estimation
- Resume interrupted runs
"""

import os
import time
import json
from pathlib import Path
import random
from datetime import datetime, timezone
from tqdm import tqdm

try:
    from elevenlabs import ElevenLabs
    from elevenlabs.core import ApiError
except ImportError:
    print("ElevenLabs SDK not installed. Run: pip install elevenlabs")
    exit(1)

try:
    from dotenv import load_dotenv
    # Load .env from project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")
except ImportError:
    pass  # dotenv is optional

# ElevenLabs voice IDs (pre-made voices)
# https://elevenlabs.io/docs/voices/premade-voices
VOICES = [
    "21m00Tcm4TlvDq8ikWAM",  # Rachel - calm, warm female
    "29vD33N1CtxCmqQRPOHJ",  # Drew - confident male
    "2EiwWnXFnvU5JabPnv8n",  # Clyde - gruff male
    "5Q0t7uMcjvnagumLfvZi",  # Paul - ground news male
    "AZnzlk1XvdvUeBnXmlld",  # Domi - strong female
    "CYw3kZ02Hs0563khs1Fj",  # Dave - conversational male
    "D38z5RcWu1voky8WS1ja",  # Fin - Irish male
    "EXAVITQu4vr4xnSDxMaL",  # Sarah - soft female
    "ErXwobaYiN019PkySvjV",  # Antoni - warm male
    "GBv7mTt0atIp3Br8iCZE",  # Thomas - calm male
    "IKne3meq5aSn9XLyUdCD",  # Charlie - casual Australian male
    "JBFqnCBsd6RMkjVDRZzb",  # George - warm British male
    "MF3mGyEYCl7XYWbV9V6O",  # Elli - emotional female
    "N2lVS1w4EtoT3dr4eOWO",  # Callum - intense male
    "TX3LPaxmHKxFdv7VOQHJ",  # Liam - articulate male
    "XB0fDUnXU5powFXDhCwa",  # Charlotte - seductive female
    "XrExE9yKIg1WjnnlVkGX",  # Matilda - warm female
    "Yko7PKs6oLEQPIk4PzMf",  # Adam - American male
    "ZQe5CZNOzWyzPSCn5a3c",  # James - Australian male
    "pFZP5JQG7iQjIQuC4Bku",  # Lily - warm British female
    "pNInz6obpgDQGcFmaJgB",  # Adam - American male (default)
    "onwK4e9ZLuTAKqWW03F9",  # Daniel - deep male
    "t0jbNlBVZ17f02VDIeMI",  # Dorothy - warm older female
    "z9fAnlkpzviPz146aGWa",  # Glinda - witch-like female
]

# Diverse neutral phrases for TTS generation
# Categories: Conversational, Professional, Informational, Storytelling,
#             Emotional, Questions, Technical, Literary

DIVERSE_PHRASES = [
    # Conversational (8 phrases)
    "Hey, I was thinking we could grab coffee this weekend if you're free.",
    "It's been ages since we last caught up. How have you been doing lately?",
    "I just saw the funniest video online and had to share it with you.",
    "Do you remember that restaurant we went to last summer? I want to go back.",
    "I'm thinking about redecorating my living room. Any suggestions?",
    "My neighbor just got the cutest puppy. You should come meet him.",
    "I finally finished that book you recommended. It was really good.",
    "What do you think about trying that new hiking trail this weekend?",

    # Professional (6 phrases)
    "Thank you for joining today's meeting. Let's review the agenda items.",
    "I wanted to follow up on our discussion from the previous quarter.",
    "The quarterly reports show significant improvement across all departments.",
    "Please ensure all documents are submitted before the deadline on Friday.",
    "We appreciate your continued partnership and look forward to future collaboration.",
    "The team has made excellent progress on the project milestones this month.",

    # Informational (6 phrases)
    "The weather forecast shows partly cloudy skies with temperatures in the mid-seventies.",
    "Public transportation services will be running on a modified schedule this holiday.",
    "The museum opens at nine in the morning and closes at five in the evening.",
    "Remember to bring your identification and confirmation number to the appointment.",
    "The local farmers market is open every Saturday from eight until noon.",
    "Registration for the summer program begins next Monday morning online.",

    # Storytelling (6 phrases)
    "Once upon a time, in a village by the sea, there lived an old fisherman.",
    "The journey through the mountains took three days and two nights of walking.",
    "She opened the door to find a mysterious package sitting on the doorstep.",
    "Long ago, before the city was built, this entire valley was covered in forest.",
    "The train whistle echoed through the station as passengers hurried to board.",
    "In the garden behind the house, a small fountain bubbled peacefully all day.",

    # Emotional - Happy/Excited (4 phrases)
    "I'm so excited! This is the best news I've heard all year!",
    "I can't believe it actually happened! This is absolutely wonderful!",
    "Today has been such an amazing day. Everything went perfectly.",
    "I'm thrilled to announce that we finally achieved our goal together.",

    # Emotional - Concerned/Thoughtful (4 phrases)
    "I've been thinking about this a lot and I wanted to share my thoughts.",
    "It's important that we take some time to consider all the options carefully.",
    "I understand this might be difficult, but I believe we can work through it.",
    "Sometimes the hardest decisions are the ones that matter the most to us.",

    # Questions (6 phrases)
    "What time would work best for you to meet tomorrow afternoon?",
    "Have you had a chance to look at the materials I sent over last week?",
    "Would you prefer to meet in person or would a video call be easier?",
    "Do you know if the store is open on Sundays during the summer months?",
    "How long have you been working on this particular project now?",
    "Are you planning to attend the conference next month in Chicago?",

    # Technical/Instructions (5 phrases)
    "First, make sure to save your work before proceeding to the next step.",
    "The process typically takes about fifteen to twenty minutes to complete.",
    "Please verify that all connections are secure before turning on the device.",
    "The update will install automatically when you restart your computer tonight.",
    "You can find more detailed instructions in the documentation section online.",

    # Literary/Expressive (5 phrases)
    "The autumn leaves danced gracefully in the cool evening breeze.",
    "Music has a way of touching our souls when words simply aren't enough.",
    "Every sunset reminds us that beautiful endings lead to new beginnings.",
    "The sound of rain against the window always brings me a sense of peace.",
    "In moments of silence, we often find the answers we've been searching for.",
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

# Available ElevenLabs models
MODELS = {
    "multilingual_v2": "eleven_multilingual_v2",
    "turbo_v2_5": "eleven_turbo_v2_5",
    "turbo_v2": "eleven_turbo_v2",
    "monolingual_v1": "eleven_monolingual_v1",
}

# All model IDs for multi-model generation
ALL_MODELS = list(MODELS.values())

# Voice settings presets: (name, stability, similarity_boost, style)
# - stability: 0.0 = variable/expressive, 1.0 = stable/consistent
# - similarity_boost: 0.0 = less like original, 1.0 = very similar to original
# - style: 0.0 = neutral, 1.0 = exaggerated style (only for v2+ models)
VOICE_SETTINGS_PRESETS = [
    ("stable_natural", 0.75, 0.75, 0.0),   # Default balanced
    ("expressive", 0.3, 0.75, 0.0),         # More emotion/variation
    ("stable_clone", 0.9, 0.9, 0.0),        # High fidelity clone
    ("variable", 0.2, 0.5, 0.0),            # Maximum variation
    ("stylized", 0.5, 0.75, 0.5),           # Style exaggeration
]


def get_voice_settings(preset_name: str) -> dict:
    """Get voice settings dict from preset name."""
    for name, stability, similarity, style in VOICE_SETTINGS_PRESETS:
        if name == preset_name:
            return {
                "stability": stability,
                "similarity_boost": similarity,
                "style": style,
            }
    # Default if not found
    return {"stability": 0.75, "similarity_boost": 0.75, "style": 0.0}


def fetch_available_voices(client: ElevenLabs) -> list:
    """
    Fetch all voices from ElevenLabs API with metadata.

    Returns list of dicts with voice_id, name, category, and labels.
    """
    try:
        response = client.voices.get_all()
        voices = []
        for v in response.voices:
            voice_info = {
                "voice_id": v.voice_id,
                "name": v.name,
                "category": getattr(v, 'category', 'unknown'),
                "labels": getattr(v, 'labels', {}),
            }
            voices.append(voice_info)
        return voices
    except Exception as e:
        print(f"Error fetching voices: {e}")
        return []


def filter_voices(voices: list, voice_filter: str) -> list:
    """
    Filter voices by category.

    Args:
        voices: List of voice dicts from fetch_available_voices
        voice_filter: 'premade', 'cloned', or 'all'
    """
    if voice_filter == "all":
        return voices
    elif voice_filter == "premade":
        return [v for v in voices if v.get("category") in ("premade", "generated")]
    elif voice_filter == "cloned":
        return [v for v in voices if v.get("category") == "cloned"]
    return voices


def print_voices(voices: list):
    """Print voice information in a formatted table."""
    print(f"\nFound {len(voices)} voices:\n")
    print(f"{'ID':<24} {'Name':<20} {'Category':<12} Labels")
    print("-" * 80)
    for v in sorted(voices, key=lambda x: x.get("name", "")):
        labels = v.get("labels", {})
        label_str = ", ".join(f"{k}={v}" for k, v in labels.items()) if labels else ""
        print(f"{v['voice_id']:<24} {v['name']:<20} {v.get('category', 'unknown'):<12} {label_str}")


def estimate_cost(
    num_samples: int,
    multi_model: bool,
    vary_settings: bool,
    phrases: list,
) -> dict:
    """
    Estimate generation cost/credits.

    Returns dict with total_samples, total_chars, and notes.
    """
    avg_chars = sum(len(p) for p in phrases) / len(phrases)

    total_samples = num_samples
    model_count = len(ALL_MODELS) if multi_model else 1
    settings_count = len(VOICE_SETTINGS_PRESETS) if vary_settings else 1

    # Note: multi-model and vary-settings cycle through options,
    # they don't multiply total samples
    total_chars = int(num_samples * avg_chars)

    return {
        "total_samples": total_samples,
        "total_chars": total_chars,
        "avg_chars_per_sample": int(avg_chars),
        "model_count": model_count,
        "settings_count": settings_count,
        "notes": [
            f"Will use {model_count} model(s)",
            f"Will use {settings_count} voice setting preset(s)",
            f"Samples cycle through models/settings for diversity",
        ]
    }


def generate_sample(
    client: ElevenLabs,
    text: str,
    voice_id: str,
    output_path: str,
    model: str = "eleven_turbo_v2_5",
    voice_settings: dict = None,
) -> bool:
    """
    Generate a single audio sample using ElevenLabs API.

    Args:
        client: ElevenLabs client instance
        text: Text to synthesize
        voice_id: ElevenLabs voice ID
        output_path: Path to save audio file
        model: Model ID to use
        voice_settings: Optional dict with stability, similarity_boost, style

    Returns:
        True if successful, False if rate limited
    """
    try:
        kwargs = {
            "voice_id": voice_id,
            "text": text,
            "model_id": model,
            "output_format": "mp3_44100_128",
        }

        if voice_settings:
            kwargs["voice_settings"] = voice_settings

        audio = client.text_to_speech.convert(**kwargs)

        # Write audio to file
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)

        return True
    except ApiError as e:
        if "rate" in str(e).lower() or "429" in str(e):
            # Rate limit - wait and retry
            time.sleep(5)
            return False
        raise


def save_metadata(
    output_path: str,
    voice_id: str,
    voice_name: str,
    model: str,
    text: str,
    voice_settings: dict = None,
):
    """Save metadata JSON alongside audio file."""
    metadata = {
        "voice_id": voice_id,
        "voice_name": voice_name,
        "model": model,
        "text": text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    if voice_settings:
        metadata["stability"] = voice_settings.get("stability")
        metadata["similarity_boost"] = voice_settings.get("similarity_boost")
        metadata["style"] = voice_settings.get("style")

    # Save as .json with same base name
    json_path = Path(output_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)


def generate_samples(
    output_dir: str,
    num_samples: int = 50,
    model: str = "eleven_turbo_v2_5",
    multi_model: bool = False,
    vary_settings: bool = False,
    save_metadata_flag: bool = False,
    resume: bool = False,
    voice_ids: list = None,
    voice_names: dict = None,
):
    """
    Generate fake samples using ElevenLabs TTS.

    Args:
        output_dir: Directory to save generated audio
        num_samples: Number of samples to generate
        model: Model ID (ignored if multi_model=True)
        multi_model: Cycle through all models
        vary_settings: Cycle through voice settings presets
        save_metadata_flag: Save JSON metadata per sample
        resume: Skip existing files
        voice_ids: List of voice IDs to use (default: VOICES)
        voice_names: Dict mapping voice_id -> name for metadata
    """
    # Check for API key
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY environment variable not set.")
        print("Set it with: export ELEVENLABS_API_KEY=your_key_here")
        print("Or create a .env file with: ELEVENLABS_API_KEY=your_key_here")
        return

    # Initialize client
    client = ElevenLabs(api_key=api_key)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_phrases = DIVERSE_PHRASES + NORMAL_PHRASES
    voices_to_use = voice_ids or VOICES
    voice_name_map = voice_names or {}

    # Determine models to use
    models_to_use = ALL_MODELS if multi_model else [model]

    # Determine settings to use
    settings_to_use = VOICE_SETTINGS_PRESETS if vary_settings else [("default", 0.75, 0.75, 0.0)]

    print(f"Generating {num_samples} fake audio samples with ElevenLabs...")
    print(f"Models: {', '.join(models_to_use)}")
    print(f"Voice settings presets: {len(settings_to_use)}")
    print(f"Voices: {len(voices_to_use)}")
    print(f"Output: {output_dir}")
    if resume:
        print("Resume mode: skipping existing files")

    generated = 0
    skipped = 0
    i = 0
    model_idx = 0
    settings_idx = 0

    pbar = tqdm(total=num_samples)

    while generated < num_samples:
        output_file = output_path / f"elevenlabs_{generated:04d}.mp3"

        # Check for resume
        if resume and output_file.exists():
            skipped += 1
            generated += 1
            pbar.update(1)
            i += 1
            continue

        text = all_phrases[i % len(all_phrases)]
        voice_id = voices_to_use[i % len(voices_to_use)]
        voice_name = voice_name_map.get(voice_id, "unknown")

        # Cycle through models
        current_model = models_to_use[model_idx % len(models_to_use)]

        # Cycle through settings
        preset_name, stability, similarity, style = settings_to_use[settings_idx % len(settings_to_use)]
        current_settings = {
            "stability": stability,
            "similarity_boost": similarity,
            "style": style,
        }

        try:
            success = generate_sample(
                client,
                text,
                voice_id,
                str(output_file),
                model=current_model,
                voice_settings=current_settings,
            )

            if success:
                # Save metadata if requested
                if save_metadata_flag:
                    save_metadata(
                        str(output_file),
                        voice_id,
                        voice_name,
                        current_model,
                        text,
                        current_settings,
                    )

                generated += 1
                pbar.update(1)
                i += 1
                model_idx += 1
                settings_idx += 1
            else:
                # Rate limited, retry same sample
                print(f"\nRate limited, waiting...")
                time.sleep(5)

        except ApiError as e:
            print(f"\nAPI Error on sample {generated}: {e}")
            i += 1  # Skip this phrase
            continue
        except Exception as e:
            print(f"\nError generating sample {generated}: {e}")
            i += 1
            continue

    pbar.close()

    if skipped > 0:
        print(f"\nSkipped {skipped} existing files (resume mode)")
    print(f"Generated {generated - skipped} new samples saved to {output_dir}")
    print("Note: Files are MP3. Run degrade_to_phone.py to convert and degrade for training.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate fake samples using ElevenLabs TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python generate_elevenlabs_samples.py --output data/raw/fake/elevenlabs --num 50

  # Maximum diversity run
  python generate_elevenlabs_samples.py --num 500 --multi-model --vary-settings --metadata

  # See available voices
  python generate_elevenlabs_samples.py --discover-voices

  # Cost estimate before generating
  python generate_elevenlabs_samples.py --num 1000 --dry-run

  # Resume interrupted generation
  python generate_elevenlabs_samples.py --num 500 --resume
"""
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/fake/elevenlabs",
        help="Output directory for generated samples",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=50,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="eleven_turbo_v2_5",
        choices=list(MODELS.values()),
        help="ElevenLabs model to use (ignored if --multi-model)",
    )

    # New options
    parser.add_argument(
        "--discover-voices",
        action="store_true",
        help="List all available voices and exit",
    )
    parser.add_argument(
        "--voice-filter",
        type=str,
        default="premade",
        choices=["premade", "cloned", "all"],
        help="Filter voices: 'premade', 'cloned', or 'all' (default: premade)",
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Generate with all 4 models for maximum diversity",
    )
    parser.add_argument(
        "--vary-settings",
        action="store_true",
        help="Vary stability/similarity across samples",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Save JSON metadata per sample",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show generation plan without making API calls",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing files (useful for interrupted runs)",
    )
    parser.add_argument(
        "--use-api-voices",
        action="store_true",
        help="Use voices from API instead of hardcoded list (enables Creator voices)",
    )

    args = parser.parse_args()

    # Check for API key early
    api_key = os.environ.get("ELEVENLABS_API_KEY")

    # Handle discover-voices
    if args.discover_voices:
        if not api_key:
            print("Error: ELEVENLABS_API_KEY required for --discover-voices")
            return
        client = ElevenLabs(api_key=api_key)
        voices = fetch_available_voices(client)
        voices = filter_voices(voices, args.voice_filter)
        print_voices(voices)
        return

    # Handle dry-run
    if args.dry_run:
        all_phrases = DIVERSE_PHRASES + NORMAL_PHRASES
        estimate = estimate_cost(
            args.num,
            args.multi_model,
            args.vary_settings,
            all_phrases,
        )
        print("\n=== DRY RUN - Generation Plan ===\n")
        print(f"Output directory: {args.output}")
        print(f"Total samples to generate: {estimate['total_samples']}")
        print(f"Average characters per sample: {estimate['avg_chars_per_sample']}")
        print(f"Total characters (estimated): {estimate['total_chars']}")
        print(f"\nDiversity settings:")
        for note in estimate['notes']:
            print(f"  - {note}")

        # Show voice count
        if args.use_api_voices and api_key:
            client = ElevenLabs(api_key=api_key)
            voices = fetch_available_voices(client)
            voices = filter_voices(voices, args.voice_filter)
            print(f"\nVoices available: {len(voices)} (from API, filter={args.voice_filter})")
        else:
            print(f"\nVoices available: {len(VOICES)} (hardcoded)")

        print(f"Phrases available: {len(all_phrases)}")
        print(f"\nOptions:")
        print(f"  --multi-model: {args.multi_model}")
        print(f"  --vary-settings: {args.vary_settings}")
        print(f"  --metadata: {args.metadata}")
        print(f"  --resume: {args.resume}")
        print(f"  --use-api-voices: {args.use_api_voices}")
        print("\nTo actually generate, remove --dry-run flag.")
        return

    # Fetch voice names for metadata if using dynamic voices or metadata flag
    voice_names = {}
    voice_ids = VOICES  # default

    if args.use_api_voices and api_key:
        # Use voices from API (includes Creator voices)
        client = ElevenLabs(api_key=api_key)
        voices = fetch_available_voices(client)
        voices = filter_voices(voices, args.voice_filter)
        voice_ids = [v["voice_id"] for v in voices]
        voice_names = {v["voice_id"]: v["name"] for v in voices}
        print(f"Using {len(voice_ids)} voices from API ({args.voice_filter})")
    elif args.metadata and api_key:
        # Fetch voice info for metadata only
        client = ElevenLabs(api_key=api_key)
        voices = fetch_available_voices(client)
        voice_names = {v["voice_id"]: v["name"] for v in voices}

    generate_samples(
        args.output,
        args.num,
        args.model,
        multi_model=args.multi_model,
        vary_settings=args.vary_settings,
        save_metadata_flag=args.metadata,
        resume=args.resume,
        voice_ids=voice_ids,
        voice_names=voice_names,
    )


if __name__ == "__main__":
    main()
