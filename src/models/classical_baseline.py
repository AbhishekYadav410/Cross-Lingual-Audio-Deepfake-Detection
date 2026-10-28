"""Classical feature-based baseline models (Model A).

Implements:
1. Support Vector Machine (RBF kernel) with CalibratedClassifierCV.
2. L2-regularized Logistic Regression.
"""

from typing import Dict, Any, Optional
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ClassicalBaseline:
    """Wrapper for classical classifiers trained on aggregated 1D spectral summaries."""

    def __init__(self, model_type: str = "svm_rbf", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self._init_pipeline()

    def _init_pipeline(self):
        if self.model_type == "svm_rbf":
            base_svc = SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                random_state=self.random_state,
            )
            clf = CalibratedClassifierCV(estimator=base_svc, ensemble=False)
        elif self.model_type == "logistic_regression":
            clf = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                solver="lbfgs",
            )
        else:
            raise ValueError(f"Unknown classical model type: {self.model_type}")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalBaseline":
        """Fits scaler and classifier on training data."""
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs probability of fake class [P(real), P(fake)]."""
        return self.pipeline.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Outputs binary prediction using given threshold."""
        prob_fake = self.predict_proba(X)[:, 1]
        return (prob_fake >= threshold).astype(int)

    def save(self, file_path: str):
        """Serializes trained pipeline to disk."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.pipeline, file_path)

    def load(self, file_path: str):
        """Deserializes trained pipeline from disk."""
        self.pipeline = joblib.load(file_path)
