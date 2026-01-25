# Agent Coordination

## Project Status: ANDROID APP COMPLETE

## Completed Components

### Phase 1: ML Pipeline
- [x] Phone degradation pipeline (`model/phone_augment.py`)
- [x] 8kHz dataset loader (`model/dataset.py`)
- [x] 4 CNN model architectures (`model/model.py`)
- [x] Training script (`model/train.py`)
- [x] TFLite export (`model/export_tflite.py`)
- [x] Dataset download script (`scripts/download_dataset.py`)
- [x] Batch degradation script (`scripts/degrade_to_phone.py`)
- [x] Inference/test script (`scripts/test_detection.py`)

### Phase 2: Android App
- [x] Project structure and Gradle config
- [x] TFLite classifier (`DeepfakeClassifier.kt`)
- [x] Audio recording service (`AudioAnalysisService.kt`)
- [x] Main UI activity (`MainActivity.kt`)
- [x] Real-time detection UI with waveform

## Next Steps
1. Train model: `python model/train.py`
2. Export to TFLite: `python model/export_tflite.py --checkpoint checkpoints/best.pt`
3. Copy `model.tflite` to `android/app/src/main/assets/`
4. Build APK: `cd android && ./gradlew assembleDebug`

## Notes
- All models expect 8kHz audio, 64 mel bands
- MobilePhoneNet is best for Android deployment
- Phone degradation simulates GSM codec, packet loss, noise
- Android app uses foreground service for background detection
