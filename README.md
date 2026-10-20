# Synthetic Speech Detection Prototype

A PyTorch experiment for classifying real and synthetic speech after reducing it to telephone bandwidth. The repo includes four models, data preparation and evaluation scripts, and an Android microphone prototype.

There is no trained checkpoint or TFLite model in the repo. You need to train a model before running file inference or the Android classifier. Earlier results are recorded in [the experiment notes](docs/HISTORICAL_TRAINING_NOTES.md), but the matching artifacts are missing and those results have not been reproduced.

## Python setup

Use Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Tests use generated audio and model weights, with TTS calls mocked. ONNX tests use `onnxscript`, included in the requirements. TensorFlow is only needed for the experimental TFLite conversion.

## Train and run inference

Put labeled audio in `data/raw/real/` and `data/raw/fake/`. Fake samples can be grouped by generator, such as `data/raw/fake/elevenlabs/`. Dataset preparation and optional TTS generation scripts are in `scripts/`; `download_dataset.py` explains how to obtain the supported datasets.

```sh
python model/train.py --data data/raw --epochs 50 --model mobile
python scripts/test_detection.py audio.wav --checkpoint checkpoints/best_model.pt --model mobile
```

The model choices are `phone_cnn`, `temporal`, `mobile`, and `attention`. Training writes `best_model.pt`, `final_model.pt`, and `history.json` to `checkpoints/`, with TensorBoard logs in `runs/`. Use `--save-dir` and `--log-dir` to change them.

Training adds random phone-like effects and crops. Validation and file inference use deterministic preprocessing and the first three seconds, padded if needed. If you already ran `scripts/degrade_to_phone.py`, train with `--no-degrade` to avoid processing the audio twice. The effects approximate telephone audio; they do not implement GSM or AMR codecs.

Keep speakers and voice generators separate between training and evaluation when measuring generalization. Results from a small or overlapping split can be misleading, and softmax scores are not calibrated authenticity probabilities.

Dataset imports accept regular files and directories. TAR and ZIP extraction rejects paths outside the destination, links, special files, and oversized archives. The limits are one million entries, 2 GiB per file, and 64 GiB expanded data; split larger datasets across archives.

## Audio and models

The Python pipeline uses 8 kHz mono audio, a 300–3400 Hz bandpass, three-second windows, 64 mel bands, a 512-point FFT, and a 256-sample hop. Model input is `(batch, 1, 64, 94)`; output is two logits for real and fake.

- `model/phone_augment.py`: filtering, augmentation, and mel features
- `model/dataset.py`: loading and dataset splits
- `model/model.py`: model definitions
- `model/train.py` and `model/evaluate.py`: training, metrics, and plots
- `scripts/test_detection.py`: inference from a checkpoint
- `tests/`: synthetic unit and regression tests

## Export and Android

`model/export_tflite.py` attempts PyTorch → ONNX → TensorFlow → TFLite conversion. It needs a separate, compatible TensorFlow/onnx-tf environment in addition to the core requirements. This path is experimental; a successful ONNX export does not establish that TFLite conversion or inference works.

```sh
python model/export_tflite.py --checkpoint checkpoints/best_model.pt --model mobile --output exported/model.tflite
```

The Android app captures microphone audio, not both sides of a phone call. Place a trained model at `android/app/src/main/assets/model.tflite`. Without it, the app explains what is missing and leaves recording off. Recording also requires microphone permission and pressing Start.

The classifier expects float32 input `[1, 1, 64, 94]` and float32 output `[1, 2]`. Python/Kotlin preprocessing tests compare synthetic fixtures for filtering, normalization, windowing, mel power, and decibel scaling. Real-device recording and trained-model inference still need validation.

With JDK 17 and Android SDK 34 installed, set `ANDROID_HOME` or `sdk.dir` in an untracked `android/local.properties`, then run:

```sh
cd android
./gradlew assembleDebug testDebugUnitTest
```

Regenerate fixtures with `python scripts/generate_android_feature_fixtures.py` when intentionally changing preprocessing.

## Datasets

Preparation scripts support [ASVspoof](https://www.asvspoof.org/), [LibriSpeech](https://www.openslr.org/12), LJSpeech, and generated samples from several TTS systems. Download the datasets separately and follow their usage terms.

## License

MIT
