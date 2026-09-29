import json
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "quantum_benchmark_results.json"

N_QUBITS = 4
N_LAYERS = 3
N_EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.05

np.random.seed(0)


# ============================================================
# COMMON VQC
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


@qml.qnode(dev)
def circuit(weights, x):

    qml.AngleEmbedding(
        x,
        wires=range(N_QUBITS),
        rotation="Y"
    )

    qml.StronglyEntanglingLayers(
        weights,
        wires=range(N_QUBITS)
    )

    return qml.expval(
        qml.PauliZ(0)
    )


def variational_classifier(weights, bias, x):

    return circuit(weights, x) + bias


def square_loss(labels, predictions):

    return qml.numpy.mean(
        (labels - qml.math.stack(predictions)) ** 2
    )


def cost(weights, bias, X, y):

    predictions = [
        variational_classifier(
            weights,
            bias,
            x
        )
        for x in X
    ]

    return square_loss(
        y,
        predictions
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    scores,
    disease_label
):

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    scores = np.asarray(scores).astype(float)

    # Quantum training uses labels {-1, +1}.
    # Convert them back to the original {0, 1}
    # labels before calculating classification metrics.
    if set(np.unique(y_true)).issubset({-1, 1}):
        y_true = ((y_true + 1) // 2).astype(int)

        other_label = 1 - disease_label

    # --------------------------------------------------------
    # Confusion matrix
    # Order:
    # [disease, non-disease]
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            disease_label,
            other_label
        ]
    )

    tp = int(cm[0][0])
    fn = int(cm[0][1])
    fp = int(cm[1][0])
    tn = int(cm[1][1])

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=disease_label,
        zero_division=0
    )

    specificity = recall_score(
        y_true,
        y_pred,
        pos_label=other_label,
        zero_division=0
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label=disease_label,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=disease_label,
        zero_division=0
    )

    # Scores must represent probability of disease
    if disease_label == 1:
        disease_scores = scores
    else:
        disease_scores = 1 - scores

    try:

        roc_auc = roc_auc_score(
            y_true == disease_label,
            disease_scores
        )

    except ValueError:

        roc_auc = None

    return {

        "accuracy": round(
            float(accuracy) * 100,
            2
        ),

        "sensitivity": round(
            float(sensitivity) * 100,
            2
        ),

        "specificity": round(
            float(specificity) * 100,
            2
        ),

        "precision": round(
            float(precision) * 100,
            2
        ),

        "f1": round(
            float(f1) * 100,
            2
        ),

        "roc_auc": (
            round(
                float(roc_auc) * 100,
                2
            )
            if roc_auc is not None
            else None
        ),

        "confusion_matrix": [
            [tn, fp],
            [fn, tp]
        ],

        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


# ============================================================
# TRAIN VQC
# ============================================================

def train_vqc(
    X_train,
    X_test,
    y_train,
    y_test,
    disease_label
):

    # --------------------------------------------------------
    # Convert labels from {0,1} to {-1,+1}
    # --------------------------------------------------------

    y_train_pm = qml.numpy.array(
        y_train * 2 - 1,
        requires_grad=False
    )

    y_test_pm = qml.numpy.array(
        y_test * 2 - 1,
        requires_grad=False
    )

    # --------------------------------------------------------
    # Initialize quantum parameters
    # --------------------------------------------------------

    weight_shape = (
        qml.StronglyEntanglingLayers.shape(
            n_layers=N_LAYERS,
            n_wires=N_QUBITS
        )
    )

    weights = qml.numpy.array(
        np.random.uniform(
            0,
            2 * np.pi,
            weight_shape
        ),
        requires_grad=True
    )

    bias = qml.numpy.array(
        0.0,
        requires_grad=True
    )

    optimizer = qml.NesterovMomentumOptimizer(
        stepsize=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    n_train = len(X_train)

    for epoch in range(N_EPOCHS):

        permutation = np.random.permutation(
            n_train
        )

        X_shuffled = X_train[
            permutation
        ]

        y_shuffled = y_train_pm[
            permutation
        ]

        for start in range(
            0,
            n_train,
            BATCH_SIZE
        ):

            X_batch = X_shuffled[
                start:start + BATCH_SIZE
            ]

            y_batch = y_shuffled[
                start:start + BATCH_SIZE
            ]

            weights, bias, _, _ = optimizer.step(
                cost,
                weights,
                bias,
                X_batch,
                y_batch
            )

        train_predictions = [
            variational_classifier(
                weights,
                bias,
                x
            )
            for x in X_train
        ]

        train_accuracy = np.mean(
            np.sign(
                qml.math.stack(
                    train_predictions
                )
            ) == y_train_pm
        )

        print(
            f"Epoch {epoch + 1:02d}/{N_EPOCHS} "
            f"| Train Accuracy: "
            f"{float(train_accuracy):.4f}"
        )

    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    test_predictions = [
        variational_classifier(
            weights,
            bias,
            x
        )
        for x in X_test
    ]

    raw_predictions = np.asarray(
        test_predictions
    )

    prediction_sign = np.sign(
        raw_predictions
    )

    y_pred = (
        (prediction_sign + 1) // 2
    ).astype(int)

    y_true = np.asarray(
        y_test_pm
    )

    # Convert VQC output [-1,+1]
    # into probability-like [0,1]
    probability = np.clip(
        (raw_predictions + 1) / 2,
        0,
        1
    )

    return calculate_metrics(
        y_true,
        y_pred,
        probability,
        disease_label
    )


# ============================================================
# BREAST CANCER
# ============================================================

def benchmark_breast_cancer():

    print()
    print("=" * 60)
    print("QUANTUM BENCHMARK — BREAST CANCER")
    print("=" * 60)

    data = load_breast_cancer()

    X = data.data
    y = data.target

    print(
        f"Dataset rows: {len(X)}"
    )

    # sklearn breast cancer:
    # 0 = malignant
    # 1 = benign
    #
    # Disease = malignant = 0

    disease_label = 0

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # PCA -> 4 qubits
    # --------------------------------------------------------

    pca = PCA(
        n_components=N_QUBITS
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    # --------------------------------------------------------
    # Scale to [0, pi]
    # --------------------------------------------------------

    angle_scaler = MinMaxScaler(
        feature_range=(0, np.pi)
    )

    X_train_angles = angle_scaler.fit_transform(
        X_train_pca
    )

    X_test_angles = angle_scaler.transform(
        X_test_pca
    )

    metrics = train_vqc(
        X_train_angles,
        X_test_angles,
        y_train,
        y_test,
        disease_label
    )

    result = {

        "name":
            "Quantum VQC Breast Cancer",

        "disease":
            "Breast Cancer",

        "type":
            "quantum",

        "model":
            "VQC Breast Cancer (4 qubits)",

        "dataset":
            "Wisconsin Breast Cancer",

        "dataset_rows":
            int(len(X)),

        "evaluated_rows":
            int(len(X_test)),

        **metrics
    }

    print()
    print("Breast Cancer VQC Results")
    print(result)

    return result


# ============================================================
# HEART DISEASE
# ============================================================

def load_heart_dataset():

    possible_paths = [

        PROJECT_ROOT /
        "heart_disease_uci.csv",

        PROJECT_ROOT /
        "data" /
        "heart_disease_uci.csv",

        PROJECT_ROOT /
        "heart.csv",

        PROJECT_ROOT /
        "data" /
        "heart.csv",
    ]

    dataset_path = None

    for path in possible_paths:

        if path.exists():

            dataset_path = path
            break

    if dataset_path is None:

        raise FileNotFoundError(
            "Heart dataset not found. "
            "Expected heart_disease_uci.csv "
            "or heart.csv in the project/data folder."
        )

    print(
        f"Using heart dataset: "
        f"{dataset_path}"
    )

    return pd.read_csv(
        dataset_path
    )


def benchmark_heart():

    print()
    print("=" * 60)
    print("QUANTUM BENCHMARK — HEART DISEASE")
    print("=" * 60)

    df = load_heart_dataset()

    # --------------------------------------------------------
    # Support both:
    #
    # num -> original UCI target
    # target -> already binarized target
    # --------------------------------------------------------

    if "num" in df.columns:

        y = (
            df["num"] >= 1
        ).astype(int)

        X = df.drop(
            columns=["num"]
        )

    elif "target" in df.columns:

        y = df["target"].astype(int)

        X = df.drop(
            columns=["target"]
        )

    else:

        raise ValueError(
            "Heart dataset must contain "
            "'num' or 'target'."
        )

    # --------------------------------------------------------
    # Remove non-feature columns
    # --------------------------------------------------------

    for column in [
        "id",
        "dataset"
    ]:

        if column in X.columns:

            X = X.drop(
                columns=[column]
            )

    # --------------------------------------------------------
    # Convert booleans
    # --------------------------------------------------------

    for column in [
        "fbs",
        "exang"
    ]:

        if column in X.columns:

            mode = X[column].dropna().mode()

            if len(mode) > 0:

                X[column] = X[column].fillna(
                    mode.iloc[0]
                )

            else:

                X[column] = X[column].fillna(
                    False
                )

            X[column] = X[column].astype(int)

    # --------------------------------------------------------
    # Fill numeric missing values
    # --------------------------------------------------------

    numeric_columns = [
        "trestbps",
        "chol",
        "thalch",
        "oldpeak",
        "ca"
    ]

    for column in numeric_columns:

        if column in X.columns:

            X[column] = X[column].fillna(
                X[column].median()
            )

    # --------------------------------------------------------
    # Fill categorical missing values
    # --------------------------------------------------------

    categorical_columns = [
        "sex",
        "cp",
        "restecg",
        "slope",
        "thal"
    ]

    for column in categorical_columns:

        if column in X.columns:

            mode = X[column].mode()

            if len(mode) > 0:

                X[column] = X[column].fillna(
                    mode.iloc[0]
                )

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    existing_categorical = [
        column
        for column in categorical_columns
        if column in X.columns
    ]

    X = pd.get_dummies(
        X,
        columns=existing_categorical,
        drop_first=False
    )

    # --------------------------------------------------------
    # Convert everything to numeric
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    X = X.fillna(0)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # PCA -> 4 qubits
    # --------------------------------------------------------

    pca = PCA(
        n_components=N_QUBITS
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    # --------------------------------------------------------
    # Scale to [0, pi]
    # --------------------------------------------------------

    angle_scaler = MinMaxScaler(
        feature_range=(0, np.pi)
    )

    X_train_angles = angle_scaler.fit_transform(
        X_train_pca
    )

    X_test_angles = angle_scaler.transform(
        X_test_pca
    )

    # --------------------------------------------------------
    # Train VQC
    # --------------------------------------------------------

    metrics = train_vqc(
        X_train_angles,
        X_test_angles,
        y_train.to_numpy(),
        y_test.to_numpy(),
        disease_label=1
    )

    result = {

        "name":
            "Quantum VQC Heart Disease",

        "disease":
            "Heart Disease",

        "type":
            "quantum",

        "model":
            "VQC Heart Disease (4 qubits)",

        "dataset":
            "UCI Heart Disease",

        "dataset_rows":
            int(len(X)),

        "evaluated_rows":
            int(len(X_test)),

        **metrics
    }

    print()
    print("Heart Disease VQC Results")
    print(result)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("QUANTUM MODEL BENCHMARK GENERATION")
    print("=" * 60)

    heart = benchmark_heart()

    breast = benchmark_breast_cancer()

    output = {

        "status":
            "success",

        "models": [

            heart,
            breast
        ],

        "note": (
            "Quantum benchmark metrics were "
            "calculated by running the 4-qubit "
            "VQC models on their respective "
            "held-out test splits."
        )
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print()
    print("=" * 60)
    print("QUANTUM BENCHMARK COMPLETE")
    print("=" * 60)

    print(
        f"Saved results to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":

    main()