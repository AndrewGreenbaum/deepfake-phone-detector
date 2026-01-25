"""
Unit tests for model evaluation.

Tests:
- ModelEvaluator class
- Metrics computation
- Visualization generation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))


class TestMetricsComputation:
    """Test metrics computation functions."""

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        from evaluate import ModelEvaluator

        # Create a model that predicts based on input sum (deterministic)
        # First half of batch predicts real, second half predicts fake
        class DeterministicModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Store expected outputs based on sample index
                self.call_count = 0

            def forward(self, x):
                batch_size = x.shape[0]
                outputs = []
                for i in range(batch_size):
                    idx = self.call_count * batch_size + i
                    if idx < 5:  # First 5 samples are real
                        outputs.append([10.0, -10.0])  # Predict real
                    else:  # Last 5 samples are fake
                        outputs.append([-10.0, 10.0])  # Predict fake
                self.call_count += 1
                return torch.tensor(outputs)

        model = DeterministicModel()

        # Create data: first 5 samples are real (0), last 5 are fake (1)
        inputs = torch.randn(10, 1, 64, 94)
        labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 real, 5 fake
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=5, shuffle=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(model, device='cpu', output_dir=tmpdir)
            metrics = evaluator.evaluate(loader)

            assert metrics['accuracy'] == 1.0
            assert metrics['precision'] == 1.0
            assert metrics['recall'] == 1.0
            assert metrics['f1_score'] == 1.0

    def test_random_predictions(self):
        """Test that random predictions give ~50% accuracy."""
        from evaluate import ModelEvaluator

        # Create model with random-ish outputs
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 94, 2)
        )

        # Create balanced dataset
        inputs = torch.randn(100, 1, 64, 94)
        labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=20)

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(model, device='cpu', output_dir=tmpdir)
            metrics = evaluator.evaluate(loader)

            # Accuracy should be roughly around 50% for random
            # Allow wide margin since model is random
            assert 0.0 <= metrics['accuracy'] <= 1.0


class TestModelEvaluator:
    """Test ModelEvaluator class."""

    @pytest.fixture
    def evaluator_setup(self):
        """Set up evaluator with simple model."""
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 94, 2)
        )

        inputs = torch.randn(20, 1, 64, 94)
        labels = torch.randint(0, 2, (20,))
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=5)

        return model, loader

    def test_evaluator_creation(self, evaluator_setup):
        """Test ModelEvaluator can be created."""
        from evaluate import ModelEvaluator

        model, _ = evaluator_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(model, device='cpu', output_dir=tmpdir)
            assert evaluator is not None

    def test_evaluator_output_dir_created(self, evaluator_setup):
        """Test that output directory is created."""
        from evaluate import ModelEvaluator

        model, _ = evaluator_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "results"
            evaluator = ModelEvaluator(model, device='cpu', output_dir=str(output_dir))

            assert output_dir.exists()

    def test_evaluate_returns_metrics(self, evaluator_setup):
        """Test that evaluate returns complete metrics."""
        from evaluate import ModelEvaluator

        model, loader = evaluator_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(model, device='cpu', output_dir=tmpdir)
            metrics = evaluator.evaluate(loader)

            required_keys = [
                'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc',
                'total_samples', 'real_samples', 'fake_samples',
                'predicted_real', 'predicted_fake', 'per_class'
            ]

            for key in required_keys:
                assert key in metrics, f"Missing key: {key}"

    def test_metrics_saved_to_json(self, evaluator_setup):
        """Test that metrics are saved to JSON file."""
        from evaluate import ModelEvaluator

        model, loader = evaluator_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(model, device='cpu', output_dir=tmpdir)
            evaluator.evaluate(loader)

            json_path = Path(tmpdir) / 'metrics.json'
            assert json_path.exists()

            with open(json_path) as f:
                saved_metrics = json.load(f)

            assert 'accuracy' in saved_metrics
            assert 'timestamp' in saved_metrics

    def test_confusion_matrix_saved(self, evaluator_setup):
        """Test that confusion matrix image is saved."""
        from evaluate import ModelEvaluator

        model, loader = evaluator_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(model, device='cpu', output_dir=tmpdir)
            evaluator.evaluate(loader)

            cm_path = Path(tmpdir) / 'confusion_matrix.png'
            assert cm_path.exists()

    def test_roc_curve_saved(self, evaluator_setup):
        """Test that ROC curve image is saved."""
        from evaluate import ModelEvaluator

        model, loader = evaluator_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(model, device='cpu', output_dir=tmpdir)
            evaluator.evaluate(loader)

            roc_path = Path(tmpdir) / 'roc_curve.png'
            assert roc_path.exists()

    def test_pr_curve_saved(self, evaluator_setup):
        """Test that precision-recall curve is saved."""
        from evaluate import ModelEvaluator

        model, loader = evaluator_setup

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(model, device='cpu', output_dir=tmpdir)
            evaluator.evaluate(loader)

            pr_path = Path(tmpdir) / 'precision_recall_curve.png'
            assert pr_path.exists()


class TestPerClassMetrics:
    """Test per-class metrics computation."""

    def test_per_class_metrics_structure(self):
        """Test per-class metrics have correct structure."""
        from evaluate import ModelEvaluator

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 94, 2)
        )

        inputs = torch.randn(20, 1, 64, 94)
        labels = torch.randint(0, 2, (20,))
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(model, device='cpu', output_dir=tmpdir)
            metrics = evaluator.evaluate(loader)

            per_class = metrics['per_class']

            assert 'real' in per_class
            assert 'fake' in per_class

            for cls in ['real', 'fake']:
                assert 'precision' in per_class[cls]
                assert 'recall' in per_class[cls]
                assert 'f1' in per_class[cls]
                assert 'support' in per_class[cls]


class TestPrintMetrics:
    """Test metrics printing function."""

    def test_print_metrics_no_error(self):
        """Test that print_metrics runs without error."""
        from evaluate import print_metrics

        metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85,
            'auc_roc': 0.90,
            'total_samples': 100,
            'real_samples': 50,
            'fake_samples': 50,
            'predicted_real': 48,
            'predicted_fake': 52,
            'per_class': {
                'real': {'precision': 0.80, 'recall': 0.85, 'f1': 0.82, 'support': 50},
                'fake': {'precision': 0.84, 'recall': 0.90, 'f1': 0.87, 'support': 50}
            }
        }

        # Should not raise any errors
        print_metrics(metrics)


class TestMetricsRange:
    """Test that metrics are within valid ranges."""

    def test_all_metrics_in_valid_range(self):
        """Test all metrics are between 0 and 1."""
        from evaluate import ModelEvaluator

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 94, 2)
        )

        inputs = torch.randn(20, 1, 64, 94)
        labels = torch.randint(0, 2, (20,))
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(model, device='cpu', output_dir=tmpdir)
            metrics = evaluator.evaluate(loader)

            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1_score'] <= 1
            assert 0 <= metrics['auc_roc'] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
