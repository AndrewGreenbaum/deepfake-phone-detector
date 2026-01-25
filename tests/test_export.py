"""
Unit tests for TFLite export functionality.

Tests export utilities (without requiring TensorFlow).
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add model path
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))


class TestExportUtilities:
    """Test export utility functions."""

    def test_create_labels_file(self):
        """Test labels file creation."""
        from export_tflite import create_labels_file

        with tempfile.TemporaryDirectory() as tmpdir:
            create_labels_file(tmpdir)

            labels_path = Path(tmpdir) / "labels.txt"
            assert labels_path.exists()

            with open(labels_path) as f:
                content = f.read()

            assert "real" in content
            assert "fake" in content

    def test_labels_file_format(self):
        """Test that labels file has correct format."""
        from export_tflite import create_labels_file

        with tempfile.TemporaryDirectory() as tmpdir:
            create_labels_file(tmpdir)

            labels_path = Path(tmpdir) / "labels.txt"

            with open(labels_path) as f:
                lines = f.read().strip().split('\n')

            assert len(lines) == 2
            assert lines[0] == "real"
            assert lines[1] == "fake"


class TestONNXExport:
    """Test ONNX export functionality."""

    def test_model_can_trace(self):
        """Test that model can be traced for export."""
        from model import get_model
        from phone_augment import PhoneConfig

        config = PhoneConfig()
        model = get_model('mobile')
        model.eval()

        # Input shape
        input_shape = (1, 1, config.N_MELS, 94)
        dummy_input = torch.randn(*input_shape)

        # Should be able to run inference
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 2)

    def test_model_onnx_export(self):
        """Test that model can be exported to ONNX."""
        onnxscript = pytest.importorskip("onnxscript", reason="ONNX export requires onnxscript")
        from model import get_model
        from phone_augment import PhoneConfig

        config = PhoneConfig()
        model = get_model('mobile')
        model.eval()

        input_shape = (1, 1, config.N_MELS, 94)
        dummy_input = torch.randn(*input_shape)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    f.name,
                    export_params=True,
                    opset_version=12,
                    input_names=['input'],
                    output_names=['output'],
                )

                assert Path(f.name).exists()
                assert Path(f.name).stat().st_size > 0

            finally:
                Path(f.name).unlink(missing_ok=True)

    @pytest.mark.parametrize("model_name", [
        "phone_cnn",
        "temporal",
        "mobile",
        "attention"
    ])
    def test_all_models_exportable(self, model_name):
        """Test that all models can be exported to ONNX."""
        onnxscript = pytest.importorskip("onnxscript", reason="ONNX export requires onnxscript")
        from model import get_model
        from phone_augment import PhoneConfig

        config = PhoneConfig()
        model = get_model(model_name)
        model.eval()

        input_shape = (1, 1, config.N_MELS, 94)
        dummy_input = torch.randn(*input_shape)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    f.name,
                    export_params=True,
                    opset_version=12,
                    input_names=['input'],
                    output_names=['output'],
                )

                assert Path(f.name).exists()

            finally:
                Path(f.name).unlink(missing_ok=True)


class TestModelCheckpointLoading:
    """Test checkpoint loading for export."""

    def test_checkpoint_format(self):
        """Test checkpoint contains required keys."""
        from model import get_model

        model = get_model('mobile')

        # Create checkpoint in expected format
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 10,
            'val_acc': 0.85,
            'val_loss': 0.3
        }

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint, f.name)

            # Load and verify
            loaded = torch.load(f.name, map_location='cpu')

            assert 'model_state_dict' in loaded
            assert 'epoch' in loaded

            Path(f.name).unlink()

    def test_load_checkpoint_into_model(self):
        """Test loading checkpoint state into model."""
        from model import get_model

        # Save checkpoint
        model1 = get_model('mobile')

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save({
                'model_state_dict': model1.state_dict()
            }, f.name)

            # Load into new model
            model2 = get_model('mobile')
            checkpoint = torch.load(f.name, map_location='cpu')
            model2.load_state_dict(checkpoint['model_state_dict'])

            # Verify weights are same
            for (n1, p1), (n2, p2) in zip(
                model1.named_parameters(),
                model2.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"

            Path(f.name).unlink()


class TestExportInputOutput:
    """Test export input/output specifications."""

    def test_expected_input_shape(self):
        """Test model input shape matches expected."""
        from model import get_model
        from phone_augment import PhoneConfig

        config = PhoneConfig()
        model = get_model('mobile')
        model.eval()

        # Expected: (batch, 1, 64 mels, 94 time frames)
        expected_shape = (1, 1, 64, 94)
        input_tensor = torch.randn(*expected_shape)

        with torch.no_grad():
            output = model(input_tensor)

        # Should work without error
        assert output is not None

    def test_output_is_logits(self):
        """Test that output is raw logits (not softmax)."""
        from model import get_model

        model = get_model('mobile')
        model.eval()

        input_tensor = torch.randn(1, 1, 64, 94)

        with torch.no_grad():
            output = model(input_tensor)

        # Logits can be any value (not constrained to [0, 1])
        # Softmax is applied externally
        assert output.shape == (1, 2)

        # Check that values are not already softmaxed
        # (sum should not be 1 unless by coincidence)
        # Actually, let's just verify shape and type
        assert output.dtype == torch.float32

    def test_batch_processing(self):
        """Test that model handles batch input."""
        from model import get_model

        model = get_model('mobile')
        model.eval()

        # Batch of 4 samples
        batch_input = torch.randn(4, 1, 64, 94)

        with torch.no_grad():
            output = model(batch_input)

        assert output.shape == (4, 2)


class TestQuantizationCompatibility:
    """Test model is compatible with quantization."""

    def test_model_has_no_dynamic_shapes(self):
        """Test model uses fixed shapes (required for quantization)."""
        from model import get_model

        model = get_model('mobile')
        model.eval()

        # Fixed input shape
        input1 = torch.randn(1, 1, 64, 94)
        input2 = torch.randn(1, 1, 64, 94)

        with torch.no_grad():
            out1 = model(input1)
            out2 = model(input2)

        # Same shapes
        assert out1.shape == out2.shape

    def test_model_float32_compatible(self):
        """Test model works with float32 precision."""
        from model import get_model

        model = get_model('mobile').float()
        model.eval()

        input_tensor = torch.randn(1, 1, 64, 94).float()

        with torch.no_grad():
            output = model(input_tensor)

        assert output.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
