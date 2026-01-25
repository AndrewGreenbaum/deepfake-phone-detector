# Phone Call Deepfake Detector

AI voice detection optimized for phone call quality audio (8kHz, GSM codec, noise).

## Project Overview

Detects AI-generated voices in phone calls where most detectors fail due to:
- 8kHz sample rate (vs 44kHz studio audio)
- 300Hz - 3.4kHz frequency band
- GSM/AMR codec compression
- Background noise, echo, packet loss

## Directory Structure

```
model/
├── phone_augment.py   # Phone degradation: 8kHz, bandpass, GSM, noise, packet loss
├── dataset.py         # PyTorch dataset for phone-quality mel spectrograms
├── model.py           # 4 architectures: PhoneCNN, TemporalFocusCNN, MobilePhoneNet, PhoneDeepfakeDetector
├── train.py           # Training with validation, checkpointing
└── export_tflite.py   # PyTorch -> ONNX -> TFLite

android/
├── app/src/main/java/com/deepfake/detector/
│   ├── MainActivity.kt         # UI with live detection results
│   ├── AudioAnalysisService.kt # Foreground service for 8kHz recording
│   └── DeepfakeClassifier.kt   # TFLite inference with mel spectrograms

scripts/
├── download_dataset.py      # Dataset setup instructions
├── prepare_asvspoof.py      # Extract ASVspoof 2019 LA dataset
├── degrade_to_phone.py      # Batch apply phone degradation
├── generate_bark_samples.py # Generate fakes with Bark TTS
└── test_detection.py        # CLI inference tool
```

## Key Technical Details

### Audio Processing
- Sample rate: 8000 Hz
- Mel bands: 64 (optimized for 300-3400Hz phone bandwidth)
- FFT size: 512, hop length: 128
- Input: ~1.5 second chunks -> 94 time frames

### Phone Degradation Pipeline (phone_augment.py)
1. Resample to 8kHz
2. Bandpass filter (300-3400Hz)
3. GSM codec simulation (quantization noise)
4. Additive background noise (10-30dB SNR)
5. Packet loss simulation (0-5%)
6. Echo/reverb

### Model Architectures (model.py)
- `phone_cnn`: Standard CNN, ~500K params
- `temporal`: Focus on temporal artifacts, ~800K params
- `mobile`: Depthwise separable convs, ~200K params (for Android)
- `attention`: Self-attention layers, ~1.2M params (best accuracy)

### Android App
- Kotlin with TensorFlow Lite
- Foreground service for continuous monitoring
- VOICE_COMMUNICATION audio source (8kHz)
- Real-time mel spectrogram computation on-device

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Dataset
python scripts/download_dataset.py
python scripts/prepare_asvspoof.py --zip data/downloads/LA.zip
python scripts/degrade_to_phone.py --input data/raw --output data/phone

# Train
python model/train.py --data data/phone --epochs 50 --model mobile

# Test
python scripts/test_detection.py audio.wav --checkpoint checkpoints/best.pt

# Export
python model/export_tflite.py --checkpoint checkpoints/best.pt --output android/app/src/main/assets/model.tflite

# Android build
cd android && ./gradlew assembleDebug
```

## Dataset Sources

- **ASVspoof 2019 LA**: Real (bonafide) + 17 spoofing systems
- **LibriSpeech**: Clean speech for real samples
- **Bark TTS**: Generate synthetic fakes locally

## Current Status

- [x] Phone degradation pipeline
- [x] Dataset loader
- [x] 4 model architectures
- [x] Training script
- [x] TFLite export
- [x] Android app scaffold
- [ ] Download ASVspoof dataset (in progress)
- [ ] Train and evaluate model
- [ ] Optimize for production
