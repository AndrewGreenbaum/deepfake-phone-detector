"""
Export trained model to TensorFlow Lite for Android deployment.
"""

import argparse
import os
from pathlib import Path
import torch
import numpy as np

from model import get_model
from phone_augment import PhoneConfig


def _tensorflow():
    """Keep TensorFlow optional for training and export utilities."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TFLite conversion requires TensorFlow and onnx-tf in a compatible "
            "export environment. Core training does not install these packages."
        ) from exc
    return tf


def export_to_tflite(
    checkpoint_path: str,
    model_name: str,
    output_path: str,
    quantize: bool = True
):
    """Export PyTorch model to TFLite."""

    output = Path(output_path)
    if output.suffix != '.tflite':
        raise ValueError("Output path must end in .tflite")
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    tf = _tensorflow()
    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError as exc:
        raise RuntimeError(
            "TFLite conversion requires onnx and onnx-tf compatible with the "
            "installed TensorFlow version. The conversion path is experimental."
        ) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    config = PhoneConfig()

    # Load PyTorch model
    print(f"Loading model from {checkpoint_path}")
    model = get_model(model_name)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Input shape: (batch, 1, 64 mels, 94 time frames)
    input_shape = (1, 1, config.N_MELS, 94)
    dummy_input = torch.randn(*input_shape)

    # Export to ONNX first
    onnx_path = str(output.with_suffix('.onnx'))
    print(f"Exporting to ONNX: {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )

    # Convert ONNX to TFLite
    print("Converting to TFLite...")

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    # Save as SavedModel
    saved_model_dir = str(output.with_name(output.stem + '_saved_model'))
    tf_rep.export_graph(saved_model_dir)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved TFLite model: {output_path} ({size_mb:.2f} MB)")

    # Test inference
    test_tflite(output_path, input_shape)

    return output_path


def test_tflite(tflite_path: str, input_shape: tuple):
    """Test TFLite model inference."""
    tf = _tensorflow()
    print("\nTesting TFLite model...")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input: {input_details[0]['shape']}, {input_details[0]['dtype']}")
    print(f"Output: {output_details[0]['shape']}")

    # Test inference
    test_input = np.random.randn(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    probs = tf.nn.softmax(output[0]).numpy()

    print(f"Test output: Real={probs[0]:.3f}, Fake={probs[1]:.3f}")
    print("TFLite model working!")


def create_labels_file(output_dir: str):
    """Create labels file for Android."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    labels = ["real", "fake"]
    labels_path = os.path.join(output_dir, "labels.txt")

    with open(labels_path, 'w') as f:
        f.write('\n'.join(labels))

    print(f"Created labels file: {labels_path}")


def main():
    parser = argparse.ArgumentParser(description="Export to TFLite")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="mobile",
                        choices=["phone_cnn", "temporal", "mobile", "attention"])
    parser.add_argument("--output", type=str, default="exported/model.tflite")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    try:
        export_to_tflite(
            args.checkpoint,
            args.model,
            args.output,
            quantize=not args.no_quantize
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")

    create_labels_file(str(Path(args.output).parent))
    print("\nExport complete!")


if __name__ == "__main__":
    main()
