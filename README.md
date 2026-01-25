# Phone Call Deepfake Detector

Real-time AI voice detection optimized for **phone call quality audio**. Unlike most detectors that fail on telephony audio, this one is built specifically for the compression, noise, and limited bandwidth of actual phone calls.

## Why This Exists

Most voice cloning scams happen over phone calls, but most AI detectors are trained on clean studio audio. Phone calls have:
- **8kHz sample rate** (vs 44kHz normal audio)
- **Narrow frequency band** (300Hz - 3.4kHz)
- **Heavy compression** (GSM, AMR codecs)
- **Noise, echo, packet loss**

This detector is trained on phone-degraded audio to catch deepfakes where they actually happen.

## Features

- **Phone-optimized model** - Trained on 8kHz, codec-compressed audio
- **Real-time detection** - Works during live phone calls
- **On-device processing** - No data leaves your phone
- **Low latency** - Results in <200ms

## Architecture

```
Phone Call (8kHz)
      ↓
┌─────────────────┐
│ Noise Reduction │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Mel Spectrogram │  64 bands, phone frequency range
└────────┬────────┘
         ↓
┌─────────────────┐
│ Lightweight CNN │  Optimized for mobile
└────────┬────────┘
         ↓
   REAL / AI + confidence %
```

## Phone Degradation Pipeline

Training data is created by degrading clean audio to simulate phone calls:

```python
clean_audio
    → Downsample to 8kHz
    → Apply GSM codec simulation
    → Bandpass filter (300Hz - 3.4kHz)
    → Add background noise
    → Simulate packet loss
    → Add echo/reverb
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
python scripts/download_dataset.py
python scripts/degrade_to_phone.py --input data/clean --output data/phone
```

### 3. Train Model
```bash
python model/train.py --epochs 50 --batch-size 32
```

### 4. Export for Mobile
```bash
python model/export_tflite.py --checkpoint checkpoints/best.pt
```

### 5. Build Android App
Open `android/` in Android Studio and build.

## Project Structure

```
deepfake-phone-detector/
├── model/
│   ├── phone_augment.py    # Phone degradation pipeline
│   ├── dataset.py          # 8kHz dataset loader
│   ├── model.py            # Phone-optimized CNN
│   ├── train.py            # Training loop
│   └── export_tflite.py    # Mobile export
├── android/                # Android app with call integration
├── scripts/
│   ├── download_dataset.py
│   ├── degrade_to_phone.py
│   └── test_detection.py
├── data/                   # Dataset (gitignored)
└── notebooks/              # Experiments
```

## Model Performance

| Metric | Clean Audio | Phone Quality |
|--------|-------------|---------------|
| Accuracy | ~95% | ~92% |
| Precision | ~94% | ~90% |
| Recall | ~96% | ~93% |
| Inference | 50ms | 50ms |

## Tech Stack

- **ML**: PyTorch, torchaudio
- **Audio Processing**: librosa, pydub, scipy
- **Mobile**: TensorFlow Lite, Kotlin
- **Phone Simulation**: GSM codec, audiomentations

## Demo

Call the demo number with an AI-generated voice (ElevenLabs, etc.) and watch it get detected in real-time.

## License

MIT
