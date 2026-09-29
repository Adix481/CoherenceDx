import json
import pickle
from pathlib import Path

import numpy as np
import torch

from hybrid_model_V1 import HybridQuantumClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results"


HEART_MODEL_PATH = RESULTS_DIR / "hybrid_heart_model.pt"
HEART_PREPROCESSOR_PATH = (
    RESULTS_DIR / "hybrid_heart_preprocessor.pkl"
)

BREAST_MODEL_PATH = RESULTS_DIR / "hybrid_breast_model.pt"
BREAST_PREPROCESSOR_PATH = (
    RESULTS_DIR / "hybrid_breast_preprocessor.pkl"
)


# ============================================================
# MODEL CACHE
# ============================================================

_heart_model = None
_breast_model = None

_heart_preprocessor = None
_breast_preprocessor = None


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

def load_preprocessor(path):
    """
    Load the saved preprocessing pipeline.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Hybrid preprocessor not found: {path}"
        )

    with open(path, "rb") as file:
        return pickle.load(file)


# ============================================================
# LOAD HYBRID MODEL
# ============================================================

def load_hybrid_model(model_path):
    """
    Load a trained HybridQuantumClassifier.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Hybrid model not found: {model_path}"
        )

    model = HybridQuantumClassifier()

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    # Support either a raw state_dict or a checkpoint dictionary.
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    return model


# ============================================================
# LOAD HEART MODEL
# ============================================================

def get_heart_model():

    global _heart_model

    if _heart_model is None:
        _heart_model = load_hybrid_model(
            HEART_MODEL_PATH
        )

    return _heart_model


# ============================================================
# LOAD BREAST MODEL
# ============================================================

def get_breast_model():

    global _breast_model

    if _breast_model is None:
        _breast_model = load_hybrid_model(
            BREAST_MODEL_PATH
        )

    return _breast_model


# ============================================================
# LOAD HEART PREPROCESSOR
# ============================================================

def get_heart_preprocessor():

    global _heart_preprocessor

    if _heart_preprocessor is None:
        _heart_preprocessor = load_preprocessor(
            HEART_PREPROCESSOR_PATH
        )

    return _heart_preprocessor


# ============================================================
# LOAD BREAST PREPROCESSOR
# ============================================================

def get_breast_preprocessor():

    global _breast_preprocessor

    if _breast_preprocessor is None:
        _breast_preprocessor = load_preprocessor(
            BREAST_PREPROCESSOR_PATH
        )

    return _breast_preprocessor


# ============================================================
# APPLY SAVED PREPROCESSING
# ============================================================

def transform_features(features, preprocessor):
    """
    Apply the exact preprocessing objects used during training.

    Expected preprocessor dictionary:

        {
            "scaler": StandardScaler,
            "pca": PCA,
            "minmax": MinMaxScaler
        }

    The final output contains exactly four
    quantum-ready features.
    """

    if not isinstance(preprocessor, dict):
        raise ValueError(
            "Invalid hybrid preprocessor format."
        )

    scaler = preprocessor.get("scaler")
    pca = preprocessor.get("pca")
    minmax = preprocessor.get("minmax")

    if scaler is None:
        raise ValueError(
            "Saved preprocessor is missing 'scaler'."
        )

    if pca is None:
        raise ValueError(
            "Saved preprocessor is missing 'pca'."
        )

    if minmax is None:
        raise ValueError(
            "Saved preprocessor is missing 'minmax'."
        )

    X = np.asarray(
        features,
        dtype=np.float32
    ).reshape(1, -1)

    X = scaler.transform(X)

    X = pca.transform(X)

    X = minmax.transform(X)

    X = torch.tensor(
        X,
        dtype=torch.float32
    )

    return X


# ============================================================
# RUN MODEL
# ============================================================

def run_hybrid_model(model, X):
    """
    Run inference through the hybrid quantum-classical model.
    """

    model.eval()

    with torch.no_grad():

        logits = model(X)

        probability = torch.sigmoid(
            logits
        ).item()

    prediction = (
        1
        if probability >= 0.5
        else 0
    )

    return prediction, probability


# ============================================================
# HEART DISEASE PREDICTION
# ============================================================

def predict_hybrid_heart(features):
    """
    Hybrid VQC prediction for heart disease.

    Features must already be arranged in the same
    feature order used during hybrid training.
    """

    model = get_heart_model()

    preprocessor = get_heart_preprocessor()

    X = transform_features(
        features,
        preprocessor
    )

    prediction, probability = run_hybrid_model(
        model,
        X
    )

    return {
        "prediction": prediction,

        "label": (
            "Disease"
            if prediction == 1
            else "No Disease"
        ),

        "probability": round(
            float(probability),
            4
        ),

        "model_name":
            "Hybrid Quantum-Classical VQC — Heart Disease",

        "model_type":
            "hybrid",

        "qubits": 4,
    }


# ============================================================
# BREAST CANCER PREDICTION
# ============================================================

def predict_hybrid_breast(features):
    """
    Hybrid VQC prediction for breast cancer.

    Features must already be arranged in the same
    feature order used during hybrid training.
    """

    model = get_breast_model()

    preprocessor = get_breast_preprocessor()

    X = transform_features(
        features,
        preprocessor
    )

    prediction, probability = run_hybrid_model(
        model,
        X
    )

    return {
        "prediction": prediction,

        "label": (
            "Malignant"
            if prediction == 1
            else "Benign"
        ),

        "probability": round(
            float(probability),
            4
        ),

        "model_name":
            "Hybrid Quantum-Classical VQC — Breast Cancer",

        "model_type":
            "hybrid",

        "qubits": 4,
    }


# ============================================================
# GENERIC HYBRID PREDICTION
# ============================================================

def predict_hybrid(
    disease,
    features
):
    """
    Generic hybrid prediction entry point.

    disease:
        "heart"
        "heart_disease"
        "breast"
        "breast_cancer"

    features:
        Original numerical feature vector in the
        exact order used during training.
    """

    if not isinstance(features, (list, tuple, np.ndarray)):
        raise ValueError(
            "Features must be a list, tuple, or numpy array."
        )

    disease = str(
        disease
    ).strip().lower()

    if disease in (
        "heart",
        "heart_disease",
        "heart disease",
    ):

        return predict_hybrid_heart(
            features
        )

    if disease in (
        "breast",
        "breast_cancer",
        "breast cancer",
    ):

        return predict_hybrid_breast(
            features
        )

    raise ValueError(
        f"Unsupported disease: {disease}"
    )


# ============================================================
# MODEL STATUS
# ============================================================

def get_hybrid_model_status():

    return {
        "heart": {
            "model_exists":
                HEART_MODEL_PATH.exists(),

            "preprocessor_exists":
                HEART_PREPROCESSOR_PATH.exists(),
        },

        "breast_cancer": {
            "model_exists":
                BREAST_MODEL_PATH.exists(),

            "preprocessor_exists":
                BREAST_PREPROCESSOR_PATH.exists(),
        },
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("HYBRID MODEL INFERENCE STATUS")
    print("=" * 60)

    status = get_hybrid_model_status()

    print(
        json.dumps(
            status,
            indent=2
        )
    )

    print("=" * 60)