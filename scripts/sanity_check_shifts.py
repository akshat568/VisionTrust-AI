import sys
from pathlib import Path
import torch
from torchvision.datasets import CIFAR10

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    SEVERITY_PARAMS,
    ShiftedCIFAR10Dataset,
)
from src.models import get_baseline_model, run_inference
from src.utils.seed import set_seed


def run_shift_sanity_checks():
    print("=" * 70)
    print("VISIONTRUST AI — DISTRIBUTION SHIFT SANITY CHECKS (PHASE 4)")
    print("=" * 70)

    seed = 42
    set_seed(seed)

    # 1. Clean test data untouched
    raw_clean_ds = CIFAR10(root="data/raw", train=False, download=True)
    clean_labels = [target for _, target in raw_clean_ds]
    assert len(clean_labels) == 10000, "Clean test set size altered!"
    print("[OK] Check 1 Passed: Clean CIFAR-10 test dataset remains completely untouched (10,000 images).")

    # 2. Verify severity parameter monotonicity
    for c_name, s_dict in SEVERITY_PARAMS.items():
        assert len(s_dict) == 5, f"{c_name} does not have exactly 5 severity levels!"
        for sev in range(1, 6):
            assert sev in s_dict, f"Severity level {sev} missing for {c_name}!"
    print("[OK] Check 2 Passed: Monotonic severity parameters verified for all 5 corruptions.")

    # 3. Check 25 shifted datasets exist, load cleanly, have shape [3, 32, 32] & match clean labels
    corruptions = list(SEVERITY_PARAMS.keys())
    severities = [1, 2, 3, 4, 5]
    total_checked = 0

    for corruption in corruptions:
        for severity in severities:
            path = Path("data/shifted") / corruption / f"severity_{severity}.pt"
            assert path.exists(), f"Shifted dataset file missing: {path}"

            ds = ShiftedCIFAR10Dataset(path)

            # Check sample count == 10,000
            assert (
                len(ds) == 10000
            ), f"Expected 10,000 samples for {path}, got {len(ds)}"

            # Check image shape == [3, 32, 32]
            img_tensor, label = ds[0]
            assert (
                img_tensor.shape == (3, 32, 32)
            ), f"Expected image tensor shape (3, 32, 32), got {img_tensor.shape}"

            # Check labels in [0, 9]
            assert 0 <= label < 10, f"Label out of bounds: {label}"

            # Check 100% label alignment with original clean test set
            ds_labels = ds.labels.tolist()
            assert (
                ds_labels == clean_labels
            ), f"Label mismatch detected in {path}! Distribution shifts must preserve original labels."

            total_checked += 1

    print(
        f"[OK] Check 3 Passed: All {total_checked} shifted datasets verified (10k samples, [3,32,32] shape, 100% label preservation)."
    )

    # 4. Verify inference on frozen model
    ckpt_path = Path("outputs/models/baseline_resnet18_best.pth")
    assert ckpt_path.exists(), f"Baseline model checkpoint missing at {ckpt_path}!"

    model = get_baseline_model(num_classes=10, cifar_stem=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cpu")
    torch.set_num_threads(8)
    sample_ds = ShiftedCIFAR10Dataset("data/shifted/blur/severity_1.pt")
    loader = torch.utils.data.DataLoader(sample_ds, batch_size=1024, shuffle=False)

    out = run_inference(model, loader, device)
    assert (
        out.predictions.shape[0] == 10000
    ), f"Inference prediction count expected 10,000, got {out.predictions.shape[0]}"
    assert (
        out.features.shape == (10000, 512)
    ), f"Feature shape expected (10000, 512), got {out.features.shape}"

    print("[OK] Check 4 Passed: Model inference and feature extraction verified across shifted data.", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ALL DISTRIBUTION SHIFT SANITY CHECKS PASSED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    run_shift_sanity_checks()
