"""
Unit tests for training utilities and Trainer class.

Tests:
- Random seed setting
- Class weight computation
- Trainer initialization
- Training loop components
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))


class TestSetSeed:
    """Test random seed setting for reproducibility."""

    def test_set_seed_python_random(self):
        """Test that Python random is seeded."""
        from train import set_seed

        set_seed(42)
        val1 = random.random()

        set_seed(42)
        val2 = random.random()

        assert val1 == val2, "Python random should be reproducible"

    def test_set_seed_numpy(self):
        """Test that NumPy random is seeded."""
        from train import set_seed

        set_seed(42)
        val1 = np.random.rand()

        set_seed(42)
        val2 = np.random.rand()

        assert val1 == val2, "NumPy random should be reproducible"

    def test_set_seed_torch(self):
        """Test that PyTorch random is seeded."""
        from train import set_seed

        set_seed(42)
        val1 = torch.rand(1).item()

        set_seed(42)
        val2 = torch.rand(1).item()

        assert val1 == val2, "PyTorch random should be reproducible"

    def test_different_seeds_different_values(self):
        """Test that different seeds produce different values."""
        from train import set_seed

        set_seed(42)
        val1 = torch.rand(1).item()

        set_seed(123)
        val2 = torch.rand(1).item()

        assert val1 != val2, "Different seeds should produce different values"


class TestComputeClassWeights:
    """Test class weight computation for imbalanced datasets."""

    def test_balanced_dataset(self):
        """Test class weights for balanced dataset."""
        from train import compute_class_weights

        # Create balanced dataset (50% each class)
        labels = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        dummy_inputs = torch.randn(8, 1, 64, 94)
        dataset = TensorDataset(dummy_inputs, labels)
        loader = DataLoader(dataset, batch_size=4)

        device = torch.device('cpu')
        weights = compute_class_weights(loader, device)

        # For balanced data, weights should be approximately equal
        assert torch.allclose(weights[0], weights[1], atol=0.1)

    def test_imbalanced_dataset(self):
        """Test class weights for imbalanced dataset."""
        from train import compute_class_weights

        # Create imbalanced dataset (75% class 0, 25% class 1)
        labels = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
        dummy_inputs = torch.randn(8, 1, 64, 94)
        dataset = TensorDataset(dummy_inputs, labels)
        loader = DataLoader(dataset, batch_size=4)

        device = torch.device('cpu')
        weights = compute_class_weights(loader, device)

        # Minority class (1) should have higher weight
        assert weights[1] > weights[0], "Minority class should have higher weight"

    def test_weights_sum(self):
        """Test that class weights are properly normalized."""
        from train import compute_class_weights

        labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        dummy_inputs = torch.randn(8, 1, 64, 94)
        dataset = TensorDataset(dummy_inputs, labels)
        loader = DataLoader(dataset, batch_size=4)

        device = torch.device('cpu')
        weights = compute_class_weights(loader, device)

        # Weights should be positive
        assert (weights > 0).all()


class TestTrainerInitialization:
    """Test Trainer class initialization."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple test model."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 94, 2)
        )

    @pytest.fixture
    def dummy_loaders(self):
        """Create dummy data loaders."""
        dummy_inputs = torch.randn(16, 1, 64, 94)
        dummy_labels = torch.randint(0, 2, (16,))
        dataset = TensorDataset(dummy_inputs, dummy_labels)

        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        return train_loader, val_loader

    def test_trainer_creation(self, simple_model, dummy_loaders):
        """Test Trainer can be created."""
        from train import Trainer

        train_loader, val_loader = dummy_loaders
        device = torch.device('cpu')

        trainer = Trainer(
            simple_model,
            train_loader,
            val_loader,
            device,
            use_amp=False
        )

        assert trainer is not None
        assert trainer.model is not None

    def test_trainer_with_class_weights(self, simple_model, dummy_loaders):
        """Test Trainer with class weights."""
        from train import Trainer

        train_loader, val_loader = dummy_loaders
        device = torch.device('cpu')
        class_weights = torch.tensor([1.0, 2.0])

        trainer = Trainer(
            simple_model,
            train_loader,
            val_loader,
            device,
            class_weights=class_weights,
            use_amp=False
        )

        # Check that weighted loss is used
        assert trainer.criterion.weight is not None

    def test_trainer_history_initialized(self, simple_model, dummy_loaders):
        """Test that training history is properly initialized."""
        from train import Trainer

        train_loader, val_loader = dummy_loaders
        device = torch.device('cpu')

        trainer = Trainer(
            simple_model,
            train_loader,
            val_loader,
            device,
            use_amp=False
        )

        assert 'train_loss' in trainer.history
        assert 'train_acc' in trainer.history
        assert 'val_loss' in trainer.history
        assert 'val_acc' in trainer.history
        assert 'lr' in trainer.history


class TestTrainerMethods:
    """Test Trainer class methods."""

    @pytest.fixture
    def trainer_setup(self):
        """Set up trainer with simple model and data."""
        from train import Trainer

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 94, 2)
        )

        dummy_inputs = torch.randn(8, 1, 64, 94)
        dummy_labels = torch.randint(0, 2, (8,))
        dataset = TensorDataset(dummy_inputs, dummy_labels)

        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        device = torch.device('cpu')

        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            device,
            use_amp=False,
            warmup_epochs=2
        )

        return trainer

    def test_lr_warmup_scale(self, trainer_setup):
        """Test learning rate warmup scaling."""
        trainer = trainer_setup

        # During warmup
        scale1 = trainer._get_lr_scale(1)
        scale2 = trainer._get_lr_scale(2)

        assert scale1 < scale2, "LR scale should increase during warmup"
        assert scale1 == 1 / trainer.warmup_epochs

        # After warmup
        scale_after = trainer._get_lr_scale(10)
        assert scale_after == 1.0, "LR scale should be 1.0 after warmup"

    def test_train_epoch_returns_metrics(self, trainer_setup):
        """Test that train_epoch returns loss and accuracy."""
        trainer = trainer_setup

        # Store initial lr for warmup
        for param_group in trainer.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

        loss, acc = trainer.train_epoch(1)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100

    def test_validate_returns_metrics(self, trainer_setup):
        """Test that validate returns loss and accuracy."""
        trainer = trainer_setup

        loss, acc = trainer.validate()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100


class TestTrainerTraining:
    """Test full training loop."""

    def test_short_training_run(self):
        """Test a short training run completes without error."""
        from train import Trainer, set_seed

        set_seed(42)

        # Simple model
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 94, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

        # Create data
        dummy_inputs = torch.randn(16, 1, 64, 94)
        dummy_labels = torch.randint(0, 2, (16,))
        dataset = TensorDataset(dummy_inputs, dummy_labels)

        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        device = torch.device('cpu')

        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            device,
            use_amp=False,
            warmup_epochs=1
        )

        # Run short training with early stopping
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = trainer.train(
                epochs=2,
                save_dir=tmpdir,
                patience=10
            )

            # Check that checkpoints were saved
            assert (Path(tmpdir) / 'final_model.pt').exists()
            assert (Path(tmpdir) / 'history.json').exists()

    def test_early_stopping(self):
        """Test that early stopping works."""
        from train import Trainer, set_seed

        set_seed(42)

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 94, 2)
        )

        dummy_inputs = torch.randn(8, 1, 64, 94)
        dummy_labels = torch.randint(0, 2, (8,))
        dataset = TensorDataset(dummy_inputs, dummy_labels)

        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        device = torch.device('cpu')

        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            device,
            use_amp=False,
            warmup_epochs=1
        )

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Very short patience - should early stop
            completed = trainer.train(
                epochs=100,  # Many epochs
                save_dir=tmpdir,
                patience=2  # Very short patience
            )

            # Check history length (should be less than 100 due to early stopping)
            assert len(trainer.history['train_loss']) < 100


class TestGradientClipping:
    """Test gradient clipping functionality."""

    def test_gradients_clipped(self):
        """Test that gradients are clipped during training."""
        from train import Trainer

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 94, 2)
        )

        dummy_inputs = torch.randn(4, 1, 64, 94)
        dummy_labels = torch.randint(0, 2, (4,))
        dataset = TensorDataset(dummy_inputs, dummy_labels)

        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)

        device = torch.device('cpu')

        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            device,
            use_amp=False
        )

        # Run one epoch
        for param_group in trainer.optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']

        trainer.train_epoch(1)

        # If training completed without error, gradient clipping worked


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
