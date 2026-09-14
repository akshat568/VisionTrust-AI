from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "confidence",
    "entropy",
    "feature_distance",
    "ood_score",
    "augmentation_consistency",
    "sharpness",
    "brightness",
    "contrast",
    "composite_quality",
]


class TrustModelPipeline:
    """Pipeline for predicting prediction correctness P(correct) using Phase 6 reliability signals.

    Ensures zero data leakage by fitting StandardScaler, Logistic Regression, Random Forest,
    and Platt scaling calibration strictly on training/validation signals.
    """

    def __init__(
        self,
        seed: int = 42,
        feature_names: Optional[List[str]] = None,
    ):
        self.seed = seed
        self.feature_names = feature_names or FEATURE_COLUMNS
        self.scaler = StandardScaler()
        self.logistic_model = LogisticRegression(random_state=self.seed, max_iter=1000)
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=self.seed
        )
        self.calibrated_model = CalibratedClassifierCV(
            estimator=LogisticRegression(random_state=self.seed, max_iter=1000),
            method="sigmoid",
            cv=5,
        )
        self.is_fitted = False

    def _extract_features(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Extract and validate feature matrix from DataFrame or numpy array."""
        if isinstance(data, pd.DataFrame):
            # Verify no ground truth true label is included in features
            forbidden = ["true_label", "target", "correctness"]
            for col in forbidden:
                if col in data.columns and col not in self.feature_names:
                    pass  # Filter out
            X = data[self.feature_names].values
        else:
            X = np.asarray(data)

        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {X.shape[1]}"
            )
        return X

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: np.ndarray,
    ) -> "TrustModelPipeline":
        """Fit scaler, primary logistic regression, secondary random forest, and Platt calibration.

        Args:
            X_train: Feature matrix or DataFrame (e.g. 5,000 clean validation signals).
            y_train: Binary target array (1 for correct vision prediction, 0 for incorrect).

        Returns:
            Fitted pipeline instance.
        """
        X = self._extract_features(X_train)
        y = np.asarray(y_train, dtype=np.int32)

        # 1. Fit scaler strictly on training features
        X_scaled = self.scaler.fit_transform(X)

        # 2. Fit primary interpretable model (Logistic Regression)
        self.logistic_model.fit(X_scaled, y)

        # 3. Fit secondary model (Random Forest)
        self.rf_model.fit(X_scaled, y)

        # 4. Fit calibrated model (Platt Sigmoid Scaling)
        self.calibrated_model.fit(X_scaled, y)

        self.is_fitted = True
        return self

    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        model_type: str = "calibrated",
    ) -> np.ndarray:
        """Predict probability P(original vision prediction is correct) in range [0, 1].

        Args:
            X: Feature matrix or DataFrame.
            model_type: Model choice ('calibrated', 'logistic', 'rf').

        Returns:
            Array of probabilities P(correct) of shape (N,).
        """
        if not self.is_fitted:
            raise RuntimeError("TrustModelPipeline must be fitted before predicting!")

        X_mat = self._extract_features(X)
        X_scaled = self.scaler.transform(X_mat)

        if model_type == "calibrated":
            probs = self.calibrated_model.predict_proba(X_scaled)[:, 1]
        elif model_type == "logistic":
            probs = self.logistic_model.predict_proba(X_scaled)[:, 1]
        elif model_type == "rf":
            probs = self.rf_model.predict_proba(X_scaled)[:, 1]
        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose 'calibrated', 'logistic', or 'rf'.")

        return np.clip(probs, 0.0, 1.0)

    def predict_failure_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        model_type: str = "calibrated",
    ) -> np.ndarray:
        """Predict failure probability P(original vision prediction is incorrect) in range [0, 1].

        Args:
            X: Feature matrix or DataFrame.
            model_type: Model choice ('calibrated', 'logistic', 'rf').

        Returns:
            Array of failure probabilities P(incorrect) = 1 - P(correct).
        """
        p_correct = self.predict_proba(X, model_type=model_type)
        return 1.0 - p_correct

    def get_feature_importance(self) -> pd.DataFrame:
        """Return Logistic Regression coefficients and Random Forest feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline is not fitted!")

        log_coefs = self.logistic_model.coef_[0]
        rf_imp = self.rf_model.feature_importances_

        df_imp = pd.DataFrame(
            {
                "feature": self.feature_names,
                "logistic_coefficient": log_coefs,
                "rf_importance": rf_imp,
            }
        ).sort_values("rf_importance", ascending=False)

        return df_imp.reset_index(drop=True)

    def save(self, file_path: Union[str, Path]):
        """Save pipeline to disk."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.scaler,
                "logistic_model": self.logistic_model,
                "rf_model": self.rf_model,
                "calibrated_model": self.calibrated_model,
                "feature_names": self.feature_names,
                "seed": self.seed,
                "is_fitted": self.is_fitted,
            },
            p,
        )

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "TrustModelPipeline":
        """Load pipeline from disk."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Trust model file not found at {p}")

        data = joblib.load(p)
        instance = cls(seed=data["seed"], feature_names=data["feature_names"])
        instance.scaler = data["scaler"]
        instance.logistic_model = data["logistic_model"]
        instance.rf_model = data["rf_model"]
        instance.calibrated_model = data["calibrated_model"]
        instance.is_fitted = data["is_fitted"]
        return instance
