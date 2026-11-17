#!/usr/bin/env python3
"""Export Trained ResNet-18 Vision Model to ONNX FP32 format.

Exports the baseline CIFAR-10 ResNet-18 model to an ONNX graph with two output heads:
  1. 'logits': Classification logits [batch_size, 10]
  2. 'features': 512-dimensional penultimate feature representation [batch_size, 512]

This dual-output ONNX model enables pure ONNX Runtime inference without PyTorch,
preserving full feature extraction for Phase 6 reliability signals (feature distance).
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None

from src.models.resnet import ResNet18CIFAR, get_baseline_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visiontrust.export_onnx")


class DualOutputResNet(nn.Module):
    """Wrapper exposing both classification logits and 512-D penultimate features."""

    def __init__(self, model: ResNet18CIFAR):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        logits, features = self.model.extract_features(x)
        return logits, features


def export_resnet18_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset_version: int = 14,
    verify_numerical_parity: bool = True,
    atol: float = 1e-5,
) -> Path:
    """Export ResNet-18 checkpoint to dual-output ONNX model.

    Args:
        checkpoint_path: Path to PyTorch .pth checkpoint.
        output_path: Destination path for .onnx file.
        opset_version: ONNX opset version (default: 14).
        verify_numerical_parity: If True, validates logits and features against PyTorch.
        atol: Maximum allowed absolute tolerance for numerical verification.

    Returns:
        Path to exported ONNX file.
    """
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading PyTorch checkpoint from '{checkpoint_path}'...")

    # Load baseline architecture and weights
    model = get_baseline_model(num_classes=10, cifar_stem=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # Wrap model for dual outputs
    wrapped_model = DualOutputResNet(model)
    wrapped_model.eval()

    # Dummy tensor for tracing: [batch_size=1, channels=3, height=32, width=32]
    dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)

    logger.info(f"Exporting dual-output model to ONNX at '{output_path}'...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits", "features"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "features": {0: "batch_size"},
        },
        opset_version=opset_version,
        dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX export successful! Model size: {size_mb:.2f} MB")

    if onnx is not None:
        logger.info("Validating ONNX model structure with onnx.checker...")
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX structural validation PASSED.")

    if verify_numerical_parity:
        if ort is None:
            logger.warning("onnxruntime not installed; skipping numerical verification.")
        else:
            logger.info("Verifying numerical parity between PyTorch and ONNX Runtime...")
            session = ort.InferenceSession(
                str(output_path),
                providers=["CPUExecutionProvider"],
            )

            # Test on multiple batch sizes
            for test_batch_size in [1, 4, 16]:
                np.random.seed(42 + test_batch_size)
                test_input_np = np.random.randn(test_batch_size, 3, 32, 32).astype(np.float32)
                test_input_torch = torch.from_numpy(test_input_np)

                with torch.no_grad():
                    py_logits, py_features = wrapped_model(test_input_torch)
                    py_logits = py_logits.numpy()
                    py_features = py_features.numpy()

                ort_outputs = session.run(
                    ["logits", "features"],
                    {"input": test_input_np},
                )
                ort_logits, ort_features = ort_outputs[0], ort_outputs[1]

                max_diff_logits = float(np.max(np.abs(py_logits - ort_logits)))
                max_diff_features = float(np.max(np.abs(py_features - ort_features)))

                logger.info(
                    f"Batch {test_batch_size:2d} | Max diff logits: {max_diff_logits:.6e} | "
                    f"Max diff features: {max_diff_features:.6e}"
                )

                if max_diff_logits > atol or max_diff_features > atol:
                    raise AssertionError(
                        f"Numerical divergence exceeded tolerance ({atol}): "
                        f"logits diff={max_diff_logits:.6e}, features diff={max_diff_features:.6e}"
                    )

            logger.info(f"Numerical parity PASSED for all tested batches within atol={atol}!")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export VisionTrust ResNet-18 baseline to dual-output ONNX FP32 format."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/models/baseline_resnet18_best.pth",
        help="Path to PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/models/baseline_resnet18.onnx",
        help="Output destination path for ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable numerical verification against PyTorch.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for numerical parity verification (default: 1e-5).",
    )

    args = parser.parse_args()
    try:
        export_resnet18_to_onnx(
            checkpoint_path=Path(args.checkpoint),
            output_path=Path(args.output),
            opset_version=args.opset,
            verify_numerical_parity=not args.no_verify,
            atol=args.atol,
        )
    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
