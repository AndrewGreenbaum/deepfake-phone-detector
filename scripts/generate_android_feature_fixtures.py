"""Generate synthetic PCM and Python-reference features for JVM parity tests."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'model'))
from phone_augment import PhoneConfig, PhoneDegradation, PhoneMelSpectrogram


def main():
    directory = Path(__file__).parents[1] / 'android/app/src/test/resources/phone_features'
    directory.mkdir(parents=True, exist_ok=True)
    config = PhoneConfig()
    t = np.arange(24000) / 8000
    long_t = np.arange(32000) / 8000
    cases = {
        'silence': np.zeros(24000, dtype=np.float32),
        'sweep': (0.5 * np.sin(2 * np.pi * (300 * t + 400 * t**2))).astype(np.float32),
        'short_noise': np.random.default_rng(42).normal(0, 0.1, 8000).astype(np.float32),
        'long_tones': (0.2 * np.sin(2 * np.pi * 600 * long_t) +
                       0.1 * np.sin(2 * np.pi * 2200 * long_t)).astype(np.float32),
    }
    for name, audio in cases.items():
        waveform = PhoneDegradation(config)(torch.from_numpy(audio).unsqueeze(0), 8000, augment=False)
        if waveform.shape[-1] < config.NUM_SAMPLES:
            waveform = torch.nn.functional.pad(waveform, (0, config.NUM_SAMPLES - waveform.shape[-1]))
        waveform = waveform[..., :config.NUM_SAMPLES]
        features = PhoneMelSpectrogram(config)(waveform).numpy()
        audio.astype('<f4').tofile(directory / f'{name}.pcm.f32')
        features.astype('<f4').tofile(directory / f'{name}.mel.f32')
        print(name, audio.shape, features.shape)


if __name__ == '__main__':
    main()
