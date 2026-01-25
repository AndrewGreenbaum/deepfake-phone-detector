# Agent Coordination

## Project Status: READY FOR TRAINING

## Completed Components
- [x] Phone degradation pipeline (`model/phone_augment.py`)
- [x] 8kHz dataset loader (`model/dataset.py`)
- [x] 4 CNN model architectures (`model/model.py`)
- [x] Training script (`model/train.py`)
- [x] TFLite export (`model/export_tflite.py`)
- [x] Dataset download script (`scripts/download_dataset.py`)
- [x] Batch degradation script (`scripts/degrade_to_phone.py`)
- [x] Inference/test script (`scripts/test_detection.py`)

## Next Steps
1. Run `pip install -r requirements.txt`
2. Run `python scripts/download_dataset.py` to setup data directories
3. Add audio samples to `data/raw/real/` and `data/raw/fake/`
4. Run `python scripts/degrade_to_phone.py` to create phone-quality versions
5. Run `python model/train.py` to train

## Notes
- All models expect 8kHz audio, 64 mel bands
- MobilePhoneNet is best for Android deployment
- Phone degradation simulates GSM codec, packet loss, noise
