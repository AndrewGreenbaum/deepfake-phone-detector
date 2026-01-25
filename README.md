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
- **4 model architectures** - From lightweight mobile to full attention-based

## Architecture

```
Phone Call (8kHz)
      |
      v
+------------------+
| Noise Reduction  |
+--------+---------+
         |
         v
+------------------+
| Mel Spectrogram  |  64 bands, phone frequency range
+--------+---------+
         |
         v
+------------------+
| Lightweight CNN  |  Optimized for mobile
+--------+---------+
         |
         v
   REAL / FAKE + confidence %
```

## Phone Degradation Pipeline

Training data is created by degrading clean audio to simulate phone calls:

```python
clean_audio
    -> Resample to 8kHz
    -> Apply GSM codec simulation
    -> Bandpass filter (300Hz - 3.4kHz)
    -> Add background noise (SNR 10-30dB)
    -> Simulate packet loss (0-5%)
    -> Add echo/reverb
```

## Quick Start

### 1. Setup Environment
```bash
cd deepfake-phone-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
# Setup directories and get instructions
python scripts/download_dataset.py

# If using ASVspoof dataset:
python scripts/prepare_asvspoof.py --zip data/downloads/LA.zip

# Apply phone degradation
python scripts/degrade_to_phone.py --input data/raw --output data/phone
```

### 3. Train Model
```bash
python model/train.py --data data/phone --epochs 50 --model mobile
```

### 4. Test Detection
```bash
python scripts/test_detection.py audio.wav --checkpoint checkpoints/best.pt --model mobile
```

### 5. Export for Mobile
```bash
python model/export_tflite.py --checkpoint checkpoints/best.pt --output android/app/src/main/assets/model.tflite
```

### 6. Build Android App
```bash
cd android
./gradlew assembleDebug
# APK: android/app/build/outputs/apk/debug/app-debug.apk
```

## Project Structure

```
deepfake-phone-detector/
├── model/
│   ├── phone_augment.py    # Phone degradation pipeline (8kHz, GSM, noise)
│   ├── dataset.py          # PyTorch dataset for phone-quality audio
│   ├── model.py            # 4 CNN architectures (PhoneCNN, Mobile, Temporal, Attention)
│   ├── train.py            # Training loop with validation
│   └── export_tflite.py    # ONNX -> TFLite conversion
├── android/
│   ├── app/src/main/
│   │   ├── java/.../       # Kotlin: MainActivity, AudioService, Classifier
│   │   └── res/            # UI layouts and resources
│   └── build.gradle
├── scripts/
│   ├── download_dataset.py   # Dataset setup instructions
│   ├── prepare_asvspoof.py   # ASVspoof dataset extraction
│   ├── degrade_to_phone.py   # Batch phone degradation
│   ├── generate_bark_samples.py  # Generate fake samples with Bark TTS
│   └── test_detection.py     # Inference script
├── data/                     # Dataset (gitignored)
│   ├── raw/real/            # Original real voice samples
│   ├── raw/fake/            # Original AI-generated samples
│   ├── phone/real/          # Phone-degraded real samples
│   └── phone/fake/          # Phone-degraded fake samples
└── requirements.txt
```

## Model Architectures

| Model | Parameters | Inference | Use Case |
|-------|-----------|-----------|----------|
| `phone_cnn` | 30K | 15ms | Ultra-lightweight baseline |
| `temporal` | 85K | 25ms | Better temporal artifacts |
| `mobile` | 17K | 10ms | Android deployment |
| `attention` | 443K | 40ms | Highest accuracy |

## Dataset Sources

### Real Voice Samples
- [ASVspoof 2019/2021](https://www.asvspoof.org/) - bonafide samples
- [LibriSpeech](https://www.openslr.org/12) - clean speech
- [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

### Fake (AI-Generated) Samples
- [ASVspoof 2019/2021](https://www.asvspoof.org/) - 17 spoofing systems
- Generate with [Bark TTS](https://github.com/suno-ai/bark) (included script)
- Generate with [Coqui TTS](https://github.com/coqui-ai/TTS)
- [ElevenLabs](https://elevenlabs.io) API

## Tech Stack

- **ML**: PyTorch 2.x, torchaudio
- **Audio Processing**: librosa, scipy, audiomentations
- **Mobile**: TensorFlow Lite, Kotlin
- **Phone Simulation**: Custom GSM codec simulation, bandpass filters

## Performance

| Dataset | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|-----|
| ASVspoof (clean) | ~95% | ~94% | ~96% | ~95% |
| ASVspoof (phone) | ~92% | ~90% | ~93% | ~91% |
| In-the-wild | ~88% | ~85% | ~90% | ~87% |

## License

MIT
