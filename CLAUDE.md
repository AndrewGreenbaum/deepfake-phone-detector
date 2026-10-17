# Development notes

Read README.md for current setup, commands, and limitations. This is a research prototype; a passing synthetic test is not evidence of deepfake-detection accuracy.

- Run `python -m pytest -q tests` after Python changes.
- Keep validation and file inference deterministic. Random telephony effects and crops belong only in training.
- Python preprocessing uses 8 kHz audio, a 300–3400 Hz bandpass, 3-second windows, 64 mel bands, a 512-point FFT, and a 256-sample hop.
- Checkpoints use `best_model.pt` and `final_model.pt` in the chosen save directory.
- TensorFlow/onnx-tf conversion is optional and experimental. Do not import TensorFlow at module load in core Python utilities.
- Android preprocessing has Python golden-fixture parity tests. Run `cd android && ./gradlew assembleDebug testDebugUnitTest` with JDK 17 / Android SDK 34. No trained model is bundled; do not describe the app as a validated live-call detector.
- Do not commit datasets, checkpoints, generated audio, or credentials.

Historical claims and experiment details are retained in `docs/HISTORICAL_TRAINING_NOTES.md`; they have not been reproduced from this checkout.
