# Earlier training experiments

These results were recorded before the October 2026 repairs. The corresponding checkpoint, split manifest, and evaluation artifacts are absent, and the earlier validation loader applied random augmentation. Treat the numbers as experiment history, not a reproduced benchmark.

| Recorded dataset | Model | Recorded train accuracy | Recorded validation accuracy |
|---|---|---|---|
| 173 samples | `phone_cnn` | 50–58% | 35–65% |
| 5,072 samples | `attention` | 99.98% | 100% |

The larger run was described as 5,000 real LJSpeech samples and 72 generated fake samples, all reduced to phone bandwidth. The notes listed automatic class balancing, 19 epochs, a single GPU, a 443K-parameter attention model, a 1.69 MB model file, and roughly 40 ms inference. The model file and timing setup were not retained.

The runs used different architectures as well as different dataset sizes, so they do not isolate the effect of adding data. The 69:1 class imbalance and missing split information also prevent a useful comparison with a held-out evaluation. Re-run with saved manifests and separate speakers and generators before drawing an accuracy or latency conclusion.
