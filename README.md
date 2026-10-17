# Phone Call Deepfake Detector

A research prototype for classifying real and synthetic speech after narrowing it to telephone bandwidth. It includes a Python training/evaluation pipeline, four PyTorch models, and an Android prototype.

No trained checkpoint or TFLite model is included. Accuracy, latency, and live-call detection have not been validated from this checkout. Historical experiment notes are retained in [docs/HISTORICAL_TRAINING_NOTES.md](docs/HISTORICAL_TRAINING_NOTES.md).

## Python setup

Use Python 3.12 in a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Tests use generated audio and model weights; TTS API calls are mocked. ONNX tests require `onnxscript`, included in the requirements. TensorFlow is not needed for training or the core tests.

## Prepare data and train

Arrange labeled audio as `data/raw/real/` and `data/raw/fake/`. Fake samples can also be grouped by TTS system, for example `data/raw/fake/elevenlabs/`. Dataset preparation scripts live in `scripts/`; `download_dataset.py` provides setup instructions.

```bash
python model/train.py --data data/raw --epochs 50 --model mobile
python scripts/test_detection.py audio.wav --checkpoint checkpoints/best_model.pt --model mobile
```

Training writes `best_model.pt`, `final_model.pt`, and `history.json` under `checkpoints/` by default, plus TensorBoard logs under `runs/`. Set `--save-dir` and `--log-dir` to change these locations.

Training applies random phone-like effects and crops. Validation, test, and file inference use fixed resampling, filtering, normalization, and the first 3 seconds (zero-padded when shorter). The filter effects approximate telephony artifacts; they do not implement GSM or AMR codecs.

For audio already processed by `scripts/degrade_to_phone.py`, pass `--no-degrade` during training to avoid applying the degradation again. Evaluation scores from small or source-overlapping splits do not establish performance on unseen speakers or voice generators.

## Models and audio representation

Available model names are `phone_cnn`, `temporal`, `mobile`, and `attention`. Compare their results on the same held-out dataset before choosing one; this repository does not establish a best architecture.

The Python pipeline uses 8 kHz mono audio, a 300–3400 Hz bandpass, 3-second windows, 64 mel bands, a 512-point FFT, and a 256-sample hop. The models receive tensors shaped `(batch, 1, 64, 94)` and return two logits: real and fake. Softmax scores are not calibrated guarantees of authenticity.

## Export and Android status

`model/export_tflite.py` contains an experimental PyTorch → ONNX → TensorFlow → TFLite path. It needs a separately configured, compatible TensorFlow/onnx-tf environment; installing the core requirements is insufficient. The script reports missing dependencies and checkpoints before writing export files.

```bash
python model/export_tflite.py --checkpoint checkpoints/best_model.pt --model mobile --output exported/model.tflite
```

A successful ONNX export does not verify TFLite conversion or trained-model inference. Android preprocessing is checked against Python-generated synthetic fixtures for filtering, normalization, windowing, mel power, and decibel scaling. The classifier requires float32 input `[1, 1, 64, 94]` and float32 output `[1, 2]`; incompatible models produce an explanation instead of starting detection.

The Android prototype captures microphone audio; it does not implement direct access to both sides of a phone call. It expects `android/app/src/main/assets/model.tflite`, which is absent. Without that asset, the app reports how to supply a model and leaves recording off. Microphone permission and the Start action are required before recording.

With JDK 17 and Android SDK 34 installed, set `ANDROID_HOME` or the `sdk.dir` entry in an untracked `android/local.properties`, then run:

```bash
cd android
./gradlew assembleDebug testDebugUnitTest
```

The JVM tests cover Python/Kotlin feature parity and missing-model startup with Robolectric. Regenerate the synthetic fixtures with `python scripts/generate_android_feature_fixtures.py` when intentionally changing preprocessing. The debug build and tests have been checked; real-device recording and trained-model inference still need validation.

## Files

| Path | Purpose |
|------|---------|
| `model/phone_augment.py` | Phone-band filtering, optional augmentation, mel transform |
| `model/dataset.py` | Audio loading and train/validation/test splits |
| `model/model.py` | Four model architectures |
| `model/train.py` | Training, gradient accumulation, logging, checkpoints |
| `model/evaluate.py` | Evaluation metrics and plots |
| `model/export_tflite.py` | Experimental mobile export |
| `scripts/test_detection.py` | File inference from a saved checkpoint |
| `scripts/` | Dataset preparation and optional TTS generators |
| `android/` | Kotlin microphone/classifier prototype |
| `tests/` | Synthetic unit and regression tests |

## Dataset references

Preparation scripts support datasets such as [ASVspoof](https://www.asvspoof.org/), [LibriSpeech](https://www.openslr.org/12), and LJSpeech, plus generated samples from several TTS systems. Obtain the datasets separately and follow their usage terms.

## License

MIT
