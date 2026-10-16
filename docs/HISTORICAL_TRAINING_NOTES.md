# Historical training notes

The following notes were present in the repository before the October 2026 repairs. They are retained for context, not as validated performance claims. The checkout has no corresponding checkpoint, split manifest, or evaluation artifacts, and the previous validation loader applied random augmentation. Reproduce the experiment with a held-out dataset before using any of these numbers.

## Training Results

### The Bitter Lesson: Scale > Tricks

Following [Richard Sutton's Bitter Lesson](http://www.incompleteideas.net/IncsightBrief/BriefLatest.html), we prioritized **data scale** over clever feature engineering. The results speak for themselves:

| Dataset Size | Model | Train Acc | Val Acc | Notes |
|-------------|-------|-----------|---------|-------|
| 173 samples | phone_cnn | 50-58% | 35-65% | Random guessing |
| **5,072 samples** | attention | **99.98%** | **100%** | Perfect classification |

**Key insight:** Scaling data 29x (173 → 5,072) transformed the model from random guessing to perfect accuracy. No architecture changes, no hyperparameter tuning - just more data.

### Training Configuration
- **Model:** Attention (443K params)
- **Data:** 5,000 real (LJSpeech) + 72 fake (TTS-generated), all phone-degraded
- **Class weights:** Automatic balancing for 69:1 imbalance
- **Epochs:** 19 (early stopping at 100% val acc)
- **Hardware:** Single GPU

### Performance Benchmarks

| Metric | Value |
|--------|-------|
| Validation Accuracy | 100.0% |
| Training Accuracy | 99.98% |
| Model Size | 1.69 MB |
| Inference Time | ~40ms |

