"""
Unit tests for model architectures.

Tests all CNN architectures for:
- Correct output shape
- Forward pass works
- Parameter counting
- Model size estimation
"""

import pytest
import torch
import sys
from pathlib import Path

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))

from model import (
    PhoneCNN,
    TemporalFocusCNN,
    MobilePhoneNet,
    PhoneDeepfakeDetector,
    SelfAttention,
    get_model,
    count_parameters,
    estimate_model_size
)


class TestModelArchitectures:
    """Test all model architectures."""

    @pytest.fixture
    def sample_input(self):
        """Create sample phone audio spectrogram input."""
        # Phone audio input: (batch, 1, 64 mels, ~94 time frames)
        batch_size = 4
        return torch.randn(batch_size, 1, 64, 94)

    def test_phone_cnn_forward(self, sample_input):
        """Test PhoneCNN forward pass."""
        model = PhoneCNN(num_classes=2)
        output = model(sample_input)

        assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"

    def test_temporal_focus_cnn_forward(self, sample_input):
        """Test TemporalFocusCNN forward pass."""
        model = TemporalFocusCNN(num_classes=2)
        output = model(sample_input)

        assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"

    def test_mobile_phone_net_forward(self, sample_input):
        """Test MobilePhoneNet forward pass."""
        model = MobilePhoneNet(num_classes=2)
        output = model(sample_input)

        assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"

    def test_phone_deepfake_detector_forward(self, sample_input):
        """Test PhoneDeepfakeDetector (attention model) forward pass."""
        model = PhoneDeepfakeDetector(num_classes=2)
        output = model(sample_input)

        assert output.shape == (4, 2), f"Expected (4, 2), got {output.shape}"


class TestSelfAttention:
    """Test self-attention module."""

    def test_attention_output_shape(self):
        """Test that attention preserves input shape."""
        channels = 128
        batch_size = 2
        height, width = 16, 23

        attention = SelfAttention(channels)
        x = torch.randn(batch_size, channels, height, width)
        output = attention(x)

        assert output.shape == x.shape, f"Attention changed shape: {x.shape} -> {output.shape}"

    def test_attention_residual_connection(self):
        """Test that attention uses residual connection (gamma starts at 0)."""
        attention = SelfAttention(64)

        # gamma should be initialized to 0
        assert attention.gamma.item() == 0.0, "gamma should start at 0"


class TestGetModel:
    """Test model factory function."""

    @pytest.mark.parametrize("model_name", [
        "phone_cnn",
        "temporal",
        "mobile",
        "attention"
    ])
    def test_get_model_valid_names(self, model_name):
        """Test get_model returns correct model for all valid names."""
        model = get_model(model_name)
        assert model is not None

        # Test forward pass
        x = torch.randn(1, 1, 64, 94)
        output = model(x)
        assert output.shape == (1, 2)

    def test_get_model_invalid_name(self):
        """Test get_model raises error for invalid name."""
        with pytest.raises(ValueError):
            get_model("invalid_model")

    def test_get_model_custom_num_classes(self):
        """Test get_model with custom number of classes."""
        model = get_model("mobile", num_classes=5)
        x = torch.randn(1, 1, 64, 94)
        output = model(x)

        assert output.shape == (1, 5), f"Expected 5 classes, got {output.shape[1]}"


class TestModelUtilities:
    """Test model utility functions."""

    def test_count_parameters(self):
        """Test parameter counting."""
        model = MobilePhoneNet()
        params = count_parameters(model)

        assert params > 0, "Model should have parameters"
        assert isinstance(params, int), "Parameter count should be integer"

    def test_estimate_model_size(self):
        """Test model size estimation."""
        model = MobilePhoneNet()
        size_mb = estimate_model_size(model)

        assert size_mb > 0, "Model size should be positive"
        assert size_mb < 50, "MobilePhoneNet should be < 50MB"

    def test_mobile_model_is_lightweight(self):
        """Test that MobilePhoneNet is actually lightweight."""
        model = MobilePhoneNet()
        size_mb = estimate_model_size(model)

        # Target is <5MB
        assert size_mb < 5, f"MobilePhoneNet should be <5MB, got {size_mb:.2f}MB"


class TestModelGradients:
    """Test that models can be trained (gradients flow)."""

    @pytest.mark.parametrize("model_name", [
        "phone_cnn",
        "temporal",
        "mobile",
        "attention"
    ])
    def test_gradients_flow(self, model_name):
        """Test that gradients flow through the model."""
        model = get_model(model_name)
        model.train()

        x = torch.randn(2, 1, 64, 94)
        target = torch.tensor([0, 1])

        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        # Check at least one parameter has gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)

        assert has_grad, f"No gradients in {model_name}"


class TestModelInference:
    """Test model inference behavior."""

    def test_eval_mode_deterministic(self):
        """Test that eval mode produces deterministic outputs."""
        model = get_model("mobile")
        model.eval()

        x = torch.randn(1, 1, 64, 94)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        assert torch.allclose(output1, output2), "Eval mode should be deterministic"

    def test_softmax_output(self):
        """Test that softmax of output sums to 1."""
        model = get_model("mobile")
        model.eval()

        x = torch.randn(4, 1, 64, 94)

        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)

        # Each row should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(4)), "Softmax should sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
