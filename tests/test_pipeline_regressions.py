"""Synthetic regression tests; no recordings, network calls, or trained weights."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parents[1] / 'model'))
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))


def test_accumulation_updates_final_partial_group(tmp_path):
    from train import Trainer
    torch.manual_seed(42)
    inputs = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
    labels = torch.tensor([0, 1, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=1)
    model = nn.Linear(2, 2)
    reference = copy.deepcopy(model)
    trainer = Trainer(model, loader, loader, torch.device('cpu'),
                      grad_accum_steps=2, warmup_epochs=0, use_amp=False,
                      log_dir=str(tmp_path / 'logs'))
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    try:
        trainer.train_epoch(1)
        for start, end in [(0, 2), (2, 3)]:
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(reference(inputs[start:end]), labels[start:end])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
            optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected)
    finally:
        trainer.writer.close()


def test_class_weights_reject_missing_class():
    from train import compute_class_weights
    loader = DataLoader(TensorDataset(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long)))
    with pytest.raises(ValueError, match='both'):
        compute_class_weights(loader, torch.device('cpu'))


def test_zero_accuracy_still_saves_first_checkpoint(tmp_path, monkeypatch):
    from train import Trainer
    loader = DataLoader(TensorDataset(torch.zeros(2, 2), torch.tensor([0, 1])), batch_size=1)
    trainer = Trainer(nn.Linear(2, 2), loader, loader, torch.device('cpu'),
                      use_amp=False, log_dir=str(tmp_path / 'logs'))
    monkeypatch.setattr(trainer, 'train_epoch', lambda epoch: (1.0, 0.0))
    monkeypatch.setattr(trainer, 'validate', lambda: (1.0, 0.0))
    trainer.train(1, save_dir=str(tmp_path / 'checkpoints'))
    assert (tmp_path / 'checkpoints' / 'best_model.pt').is_file()


@pytest.fixture
def audio_dataset(tmp_path):
    t = np.arange(32000, dtype=np.float32) / 8000
    waveform = np.sin(2 * np.pi * (300 * t + 80 * t ** 2)) * 0.5
    for label in ('real', 'fake'):
        directory = tmp_path / label
        directory.mkdir()
        for i in range(5):
            sf.write(directory / f'{i}.wav', waveform, 8000)
    return tmp_path


def test_validation_and_test_are_deterministic_and_training_stays_augmented(audio_dataset):
    from dataset import create_dataloaders
    train, val, test, _ = create_dataloaders(str(audio_dataset), batch_size=1,
                                            train_split=0.6, val_split=0.2,
                                            num_workers=0)
    assert train.dataset.dataset.augment is True
    for loader in (val, test):
        assert loader.dataset.dataset.augment is False
        first, label = loader.dataset[0]
        second, same_label = loader.dataset[0]
        assert label == same_label
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert set(train.dataset.indices).isdisjoint(val.dataset.indices)
    assert set(train.dataset.indices).isdisjoint(test.dataset.indices)


def test_inference_matches_nonaugmented_dataset(audio_dataset):
    from dataset import PhoneAudioDataset
    from phone_augment import PhoneConfig, PhoneDegradation, PhoneMelSpectrogram
    from test_detection import preprocess_audio
    config = PhoneConfig()
    dataset = PhoneAudioDataset(str(audio_dataset), augment=False)
    expected, _ = dataset[0]
    path, _ = dataset.samples[0]
    degrader = PhoneDegradation(config)
    waveform = preprocess_audio(path, config, degrader)
    second = preprocess_audio(path, config, degrader)
    assert waveform.shape == (config.NUM_SAMPLES,)
    torch.testing.assert_close(waveform, second, rtol=0, atol=0)
    actual = PhoneMelSpectrogram(config)(waveform.unsqueeze(0))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_threshold_result_score_matches_the_selected_label(audio_dataset):
    from phone_augment import PhoneConfig, PhoneDegradation, PhoneMelSpectrogram
    from test_detection import detect_deepfake

    class FixedModel(nn.Module):
        def forward(self, inputs):
            return torch.tensor([[0.3, 0.7]]).log()

    config = PhoneConfig()
    result = detect_deepfake(str(audio_dataset / 'real' / '0.wav'), FixedModel(),
                            PhoneMelSpectrogram(config), PhoneDegradation(config),
                            config, 'cpu', threshold=0.8)
    assert result['prediction'] == 'REAL'
    assert result['confidence'] == pytest.approx(result['real_probability'])


def test_export_reports_missing_optional_dependency(tmp_path, monkeypatch):
    import export_tflite
    checkpoint = tmp_path / 'model.pt'
    checkpoint.touch()
    monkeypatch.setitem(sys.modules, 'tensorflow', None)
    with pytest.raises(RuntimeError, match='TensorFlow and onnx-tf'):
        export_tflite.export_to_tflite(str(checkpoint), 'mobile', str(tmp_path / 'model.tflite'))
    assert not (tmp_path / 'model.onnx').exists()


def test_export_cli_accepts_filename_in_current_directory(tmp_path, monkeypatch):
    import export_tflite
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['export_tflite.py', '--checkpoint', 'input.pt',
                                    '--output', 'model.tflite'])
    monkeypatch.setattr(export_tflite, 'export_to_tflite', lambda *a, **k: None)
    export_tflite.main()
    assert (tmp_path / 'labels.txt').read_text() == 'real\nfake'
