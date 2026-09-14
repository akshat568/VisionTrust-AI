import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliability.trust_model import TrustModelPipeline, FEATURE_COLUMNS
from src.utils.seed import set_seed


def train_trust_model(
    val_signals_path: str = "outputs/metrics/reliability_signals_val.csv",
    output_model_path: str = "outputs/models/trust_model.pkl",
    seed: int = 42,
):
    """Train TrustModelPipeline strictly on the clean validation set."""
    val_p = Path(val_signals_path).resolve()
    model_p = Path(output_model_path).resolve()

    if not val_p.exists():
        raise FileNotFoundError(f"Validation signals missing at {val_p}. Run `scripts/extract_val_signals.py` first.")

    set_seed(seed)
    df_val = pd.read_csv(val_p)

    print("=" * 70, flush=True)
    print("VISIONTRUST AI — TRAINING TRUST MODEL (PHASE 7)", flush=True)
    print("=" * 70, flush=True)
    print(f"Training set size: {len(df_val):,} validation samples", flush=True)
    print(f"Features: {FEATURE_COLUMNS}", flush=True)
    print("Target: Binary prediction correctness (1 = Correct, 0 = Incorrect)\n", flush=True)

    # Verify no true label or target in features
    X_val = df_val[FEATURE_COLUMNS]
    y_val = df_val["correctness"].values

    pipeline = TrustModelPipeline(seed=seed, feature_names=FEATURE_COLUMNS)
    pipeline.fit(X_val, y_val)

    # Compute validation accuracy/scores
    val_p_correct = pipeline.predict_proba(X_val, model_type="calibrated")
    val_preds_binary = (val_p_correct >= 0.5).astype(int)
    val_acc = np.mean(val_preds_binary == y_val)

    print(f"[OK] Trust Model successfully trained!")
    print(f"Validation Correctness Binary Accuracy: {val_acc*100:.2f}%\n")

    # Display Feature Importance Table
    df_imp = pipeline.get_feature_importance()
    print("FEATURE IMPORTANCE / COEFFICIENTS:")
    print("-" * 55)
    for _, r in df_imp.iterrows():
        print(
            f"  {r['feature']:25s} | Logistic Coef: {r['logistic_coefficient']:+7.4f} | "
            f"RF Importance: {r['rf_importance']:6.4f}"
        )
    print("-" * 55 + "\n")

    pipeline.save(model_p)
    print(f"Trust Model saved to: {model_p}\n")


def main():
    parser = argparse.ArgumentParser(description="Train Phase 7 Trust Model.")
    parser.add_argument("--val-signals", type=str, default="outputs/metrics/reliability_signals_val.csv")
    parser.add_argument("--output-model", type=str, default="outputs/models/trust_model.pkl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_trust_model(
        val_signals_path=args.val_signals,
        output_model_path=args.output_model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
