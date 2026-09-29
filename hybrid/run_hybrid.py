"""
Hybrid Quantum-Classical Experiment Runner.

Runs the complete hybrid pipeline independently from the
classical and standalone quantum pipelines.

Pipeline:

    Dataset
        ↓
    Hybrid preprocessing
        ↓
    4-qubit VQC
        ↓
    Classical output layer
        ↓
    Evaluation
        ↓
    JSON benchmark results

Usage:

    python -m hybrid.run_hybrid

or:

    python hybrid/run_hybrid.py
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

try:
    from .hybrid_preprocessing import preprocess_hybrid
    from .hybrid_model import (
        HybridQuantumClassifier,
        train_hybrid_model,
    )
except ImportError:
    from hybrid_preprocessing import preprocess_hybrid
    from hybrid_model import (
        HybridQuantumClassifier,
        train_hybrid_model,
    )


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "Results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.01


DATASET_CONFIG = {
    "heart": {
        "name": "Hybrid Heart Disease",
        "model_name": "Hybrid Quantum-Classical VQC Heart Disease",
        "dataset_name": "UCI Heart Disease",
    },

    "wdbc": {
        "name": "Hybrid Breast Cancer",
        "model_name": "Hybrid Quantum-Classical VQC Breast Cancer",
        "dataset_name": "Wisconsin Breast Cancer",
    },
}


# ============================================================
# DATASET SELECTION
# ============================================================

def select_dataset():

    print()
    print("=" * 60)
    print("HYBRID QUANTUM-CLASSICAL MODEL")
    print("=" * 60)

    print()
    print("Available datasets:")
    print("1. Heart Disease")
    print("2. Breast Cancer")

    choice = input(
        "\nEnter dataset (heart/wdbc): "
    ).strip().lower()

    if choice not in DATASET_CONFIG:

        raise ValueError(
            f"Invalid dataset '{choice}'. "
            "Please enter 'heart' or 'wdbc'."
        )

    return choice


# ============================================================
# EVALUATION
# ============================================================

def calculate_hybrid_metrics(
    y_true,
    y_pred,
    probabilities,
):
    """
    Calculate binary classification metrics.
    """

    y_true = np.asarray(
        y_true
    ).astype(int)

    y_pred = np.asarray(
        y_pred
    ).astype(int)

    probabilities = np.asarray(
        probabilities
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    true_negative = int(
        cm[0, 0]
    )

    false_positive = int(
        cm[0, 1]
    )

    false_negative = int(
        cm[1, 0]
    )

    true_positive = int(
        cm[1, 1]
    )

    # --------------------------------------------------------
    # Classification Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    specificity = recall_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )

    except ValueError:

        roc_auc = 0.0

    return {
        "accuracy": round(
            float(accuracy * 100),
            2
        ),

        "sensitivity": round(
            float(sensitivity * 100),
            2
        ),

        "specificity": round(
            float(specificity * 100),
            2
        ),

        "precision": round(
            float(precision * 100),
            2
        ),

        "f1": round(
            float(f1 * 100),
            2
        ),

        "roc_auc": round(
            float(roc_auc * 100),
            2
        ),

        "confusion_matrix": [
            [
                true_negative,
                false_positive
            ],
            [
                false_negative,
                true_positive
            ]
        ],

        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


# ============================================================
# RUN HYBRID EXPERIMENT
# ============================================================

def run_hybrid_experiment(
    dataset
):

    config = DATASET_CONFIG[
        dataset
    ]

    print()
    print("=" * 60)
    print(
        f"HYBRID MODEL — "
        f"{config['name'].upper()}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    torch.manual_seed(0)
    np.random.seed(0)

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessing_info,
    ) = preprocess_hybrid(
        dataset
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = HybridQuantumClassifier()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("HYBRID MODEL TRAINING")
    print("=" * 60)

    training_start = time.perf_counter()

    model = train_hybrid_model(
        model,
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    print()
    print(
        f"Training Time: "
        f"{training_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("HYBRID MODEL INFERENCE")
    print("=" * 60)

    model.eval()

    inference_start = time.perf_counter()

    with torch.no_grad():

        test_logits = model(
            X_test
        )

        test_probabilities = torch.sigmoid(
            test_logits
        )

        test_predictions = (
            test_probabilities >= 0.5
        ).float()

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    y_true = y_test.numpy()

    y_pred = (
        test_predictions
        .numpy()
    )

    probabilities = (
        test_probabilities
        .numpy()
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_hybrid_metrics(
        y_true,
        y_pred,
        probabilities
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("HYBRID MODEL RESULTS")
    print("=" * 60)

    print(
        f"Accuracy    : "
        f"{metrics['accuracy']:.2f}%"
    )

    print(
        f"Sensitivity : "
        f"{metrics['sensitivity']:.2f}%"
    )

    print(
        f"Specificity : "
        f"{metrics['specificity']:.2f}%"
    )

    print(
        f"Precision   : "
        f"{metrics['precision']:.2f}%"
    )

    print(
        f"F1 Score    : "
        f"{metrics['f1']:.2f}%"
    )

    print(
        f"ROC-AUC     : "
        f"{metrics['roc_auc']:.2f}%"
    )

    print()
    print("Confusion Matrix:")

    print(
        np.array(
            metrics["confusion_matrix"]
        )
    )

    print()
    print(
        f"Inference Time: "
        f"{inference_time:.4f} seconds"
    )

    print(
        f"Inference Time/Sample: "
        f"{(inference_time / len(X_test)) * 1000:.4f} ms"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_file = (
        RESULTS_DIR
        / f"hybrid_{dataset}_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_file
    )

    # ========================================================
    # SAVE PREPROCESSOR
    # ========================================================

    preprocessor_file = (
        RESULTS_DIR
        / f"hybrid_{dataset}_preprocessor.pkl"
    )

    # Do not pickle the tensors.
    # Save only the preprocessing objects and metadata.

    preprocessor_to_save = {
        "dataset": dataset,
        "n_qubits": preprocessing_info[
            "n_qubits"
        ],
        "scaler": preprocessing_info[
            "scaler"
        ],
        "pca": preprocessing_info[
            "pca"
        ],
        "angle_scaler": preprocessing_info[
            "angle_scaler"
        ],
        "dataset_rows": preprocessing_info[
            "dataset_rows"
        ],
        "original_features": preprocessing_info[
            "original_features"
        ],
        "explained_variance": preprocessing_info[
            "explained_variance"
        ],
    }

    with open(
        preprocessor_file,
        "wb"
    ) as file:

        pickle.dump(
            preprocessor_to_save,
            file
        )

    # ========================================================
    # BUILD JSON RESULT
    # ========================================================

    result = {
        "name": config["name"],

        "disease": (
            "Heart Disease"
            if dataset == "heart"
            else "Breast Cancer"
        ),

        "type": "hybrid",

        "model": config[
            "model_name"
        ],

        "dataset": config[
            "dataset_name"
        ],

        "dataset_rows": int(
            preprocessing_info[
                "dataset_rows"
            ]
        ),

        "evaluated_rows": int(
            len(X_test)
        ),

        "training_time_seconds": round(
            float(training_time),
            4
        ),

        "inference_time_seconds": round(
            float(inference_time),
            4
        ),

        "inference_time_per_sample_ms": round(
            float(
                (
                    inference_time
                    / len(X_test)
                ) * 1000
            ),
            4
        ),

        "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "n_qubits": 4,

        "n_layers": 3,

        **metrics,
    }

    return (
        result,
        model_file,
        preprocessor_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = select_dataset()

    result, model_file, preprocessor_file = (
        run_hybrid_experiment(
            dataset
        )
    )

    # ========================================================
    # LOAD EXISTING HYBRID RESULTS
    # ========================================================

    results_file = (
        RESULTS_DIR
        / "hybrid_benchmark_results.json"
    )

    if results_file.exists():

        try:

            with open(
                results_file,
                "r",
                encoding="utf-8"
            ) as file:

                existing = json.load(
                    file
                )

        except (
            json.JSONDecodeError,
            OSError
        ):

            existing = {
                "status": "success",
                "models": []
            }

    else:

        existing = {
            "status": "success",
            "models": []
        }

    # ========================================================
    # REPLACE RESULT FOR SAME DATASET
    # ========================================================

    models = existing.get(
        "models",
        []
    )

    models = [
        model
        for model in models
        if model.get(
            "disease"
        ) != result["disease"]
    ]

    models.append(
        result
    )

    # ========================================================
    # FINAL JSON
    # ========================================================

    output = {
        "status": "success",

        "models": models,

        "note": (
            "Hybrid quantum-classical benchmark "
            "results generated by training the "
            "4-qubit hybrid VQC model on the "
            "respective held-out test split."
        ),
    }

    with open(
        results_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 60)
    print("HYBRID BENCHMARK COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Saved benchmark results:"
    )

    print(
        results_file
    )

    print()
    print(
        f"Saved model:"
    )

    print(
        model_file
    )

    print()
    print(
        f"Saved preprocessor:"
    )

    print(
        preprocessor_file
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()