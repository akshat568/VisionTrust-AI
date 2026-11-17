#!/usr/bin/env python3
"""Measure and compare actual process memory (Working Set / RSS) between PyTorch and ONNX Runtime."""

import gc
import os
import subprocess
from pathlib import Path
import sys
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_process_memory_mb() -> float:
    """Measure resident set memory of current process via PowerShell."""
    pid = os.getpid()
    out = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {pid}).WorkingSet64 / 1MB"]
    )
    return float(out.strip())


def measure_onnx_pipeline():
    print("=" * 60)
    print("MEASURING ONNX RUNTIME PIPELINE MEMORY")
    print("=" * 60)
    m0 = get_process_memory_mb()
    print(f"Base process memory: {m0:.2f} MB")

    from src.inference.onnx_pipeline import ONNXVisionTrustPipeline
    m_import = get_process_memory_mb()
    print(f"After ONNX pipeline import: {m_import:.2f} MB (+{m_import - m0:.2f} MB)")

    pipeline = ONNXVisionTrustPipeline()
    m_load = get_process_memory_mb()
    print(f"After ONNX model & artifacts loaded: {m_load:.2f} MB (+{m_load - m_import:.2f} MB)")

    img = Image.new("RGB", (32, 32))
    res = pipeline.predict_image(img)
    m_first = get_process_memory_mb()
    print(f"After 1st prediction: {m_first:.2f} MB (+{m_first - m_load:.2f} MB)")

    for _ in range(20):
        pipeline.predict_image(img)

    gc.collect()
    m_steady = get_process_memory_mb()
    print(f"After 20 predictions (steady state): {m_steady:.2f} MB")
    print(f"Total ONNX memory increase: {m_steady - m0:.2f} MB")
    print("=" * 60)
    return m_steady


if __name__ == "__main__":
    measure_onnx_pipeline()
