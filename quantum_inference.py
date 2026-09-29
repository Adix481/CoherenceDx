from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

QUANTUM_DIR = PROJECT_ROOT / "quantum_models"
QUANTUM_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# VQC CONFIGURATION
# ============================================================

N_QUBITS = 4
N_LAYERS = 3
BATCH_SIZE = 16
N_EPOCHS = 15
LEARNING_RATE = 0.05


# ============================================================
# MODEL FILES
# ============================================================

HEART_MODEL_FILE = (
    QUANTUM_DIR / "heart_vqc.npz"
)

HEART_PREPROCESSOR_FILE = (
    QUANTUM_DIR / "heart_preprocessor.npz"
)

BREAST_MODEL_FILE = (
    QUANTUM_DIR / "breast_vqc.npz"
)

BREAST_PREPROCESSOR_FILE = (
    QUANTUM_DIR / "breast_preprocessor.npz"
)


# ============================================================
# QUANTUM DEVICE
# ============================================================

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)


# ============================================================
# QUANTUM CIRCUIT
# ============================================================

@qml.qnode(
    dev,
    diff_method="backprop"
)
def circuit(
    weights,
    x
):

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


# ============================================================
# VARIATIONAL CLASSIFIER
# ============================================================

def variational_classifier(
    weights,
    bias,
    x
):

    return (
        circuit(
            weights,
            x
        )
        + bias
    )


# ============================================================
# LOSS FUNCTION
# ============================================================

def square_loss(
    labels,
    predictions
):

    return pnp.mean(
        (
            labels
            - qml.math.stack(
                predictions
            )
        ) ** 2
    )


# ============================================================
# COST FUNCTION
# ============================================================

def cost(
    weights,
    bias,
    X,
    y
):

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
# TRAIN VQC
# ============================================================

def train_vqc(
    X,
    y
):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    # --------------------------------------------------------
    # STANDARD SCALER
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=N_QUBITS
    )

    X_train_pca = (
        pca.fit_transform(
            X_train_scaled
        )
    )

    X_test_pca = (
        pca.transform(
            X_test_scaled
        )
    )

    # --------------------------------------------------------
    # ANGLE SCALER
    # --------------------------------------------------------

    angle_scaler = MinMaxScaler(
        feature_range=(0, np.pi)
    )

    X_train_angles = (
        angle_scaler.fit_transform(
            X_train_pca
        )
    )

    X_test_angles = (
        angle_scaler.transform(
            X_test_pca
        )
    )

    # --------------------------------------------------------
    # CONVERT LABELS
    # --------------------------------------------------------

    y_train_pm = pnp.array(
        y_train * 2 - 1,
        requires_grad=False
    )

    y_test_pm = pnp.array(
        y_test * 2 - 1,
        requires_grad=False
    )

    # --------------------------------------------------------
    # INITIALIZE PARAMETERS
    # --------------------------------------------------------

    np.random.seed(0)

    weight_shape = (
        qml.StronglyEntanglingLayers.shape(
            n_layers=N_LAYERS,
            n_wires=N_QUBITS
        )
    )

    weights = pnp.array(
        np.random.uniform(
            0,
            2 * np.pi,
            weight_shape
        ),
        requires_grad=True
    )

    bias = pnp.array(
        0.0,
        requires_grad=True
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    opt = qml.NesterovMomentumOptimizer(
        stepsize=LEARNING_RATE
    )

    n_train = len(
        X_train_angles
    )

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(
        N_EPOCHS
    ):

        perm = np.random.permutation(
            n_train
        )

        X_shuffled = (
            X_train_angles[
                perm
            ]
        )

        y_shuffled = (
            y_train_pm[
                perm
            ]
        )

        for start in range(
            0,
            n_train,
            BATCH_SIZE
        ):

            X_batch = (
                X_shuffled[
                    start:start + BATCH_SIZE
                ]
            )

            y_batch = (
                y_shuffled[
                    start:start + BATCH_SIZE
                ]
            )

            weights, bias, _, _ = (
                opt.step(
                    cost,
                    weights,
                    bias,
                    X_batch,
                    y_batch
                )
            )

        # ----------------------------------------------------
        # TRAINING ACCURACY
        # ----------------------------------------------------

        train_predictions = [
            variational_classifier(
                weights,
                bias,
                x
            )
            for x in X_train_angles
        ]

        train_accuracy = pnp.mean(
            pnp.sign(
                qml.math.stack(
                    train_predictions
                )
            )
            == y_train_pm
        )

        print(
            f"Epoch {epoch + 1:2d}/{N_EPOCHS} "
            f"| Train Acc: "
            f"{float(train_accuracy):.4f}"
        )

    return (
        np.asarray(weights),
        float(bias),
        scaler,
        pca,
        angle_scaler,
        X_test_angles,
        y_test
    )


# ============================================================
# SAVE VQC MODEL
# ============================================================

def save_vqc(
    model_file,
    preprocessor_file,
    weights,
    bias,
    scaler,
    pca,
    angle_scaler
):

    # --------------------------------------------------------
    # MODEL PARAMETERS
    # --------------------------------------------------------

    np.savez(
        model_file,
        weights=weights,
        bias=np.array(
            [bias]
        )
    )

    # --------------------------------------------------------
    # PREPROCESSING PARAMETERS
    # --------------------------------------------------------

    np.savez(
        preprocessor_file,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance=pca.explained_variance_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
        pca_singular_values=pca.singular_values_,
        angle_min=angle_scaler.min_,
        angle_scale=angle_scaler.scale_,
        angle_data_min=angle_scaler.data_min_,
        angle_data_max=angle_scaler.data_max_,
        angle_data_range=angle_scaler.data_range_
    )

    print(
        f"Saved model: "
        f"{model_file}"
    )

    print(
        f"Saved preprocessor: "
        f"{preprocessor_file}"
    )


# ============================================================
# LOAD VQC MODEL
# ============================================================

def load_vqc(
    model_file,
    preprocessor_file
):

    if not model_file.exists():
        return None

    if not preprocessor_file.exists():
        return None

    try:

        # ----------------------------------------------------
        # LOAD FILES
        # ----------------------------------------------------

        model_data = np.load(
            model_file
        )

        preprocessor_data = np.load(
            preprocessor_file
        )

        # ----------------------------------------------------
        # LOAD WEIGHTS
        # ----------------------------------------------------

        weights = pnp.array(
            model_data["weights"],
            requires_grad=False
        )

        bias = float(
            model_data["bias"][0]
        )

        # ----------------------------------------------------
        # STANDARD SCALER
        # ----------------------------------------------------

        scaler = StandardScaler()

        scaler.mean_ = (
            preprocessor_data[
                "scaler_mean"
            ]
        )

        scaler.scale_ = (
            preprocessor_data[
                "scaler_scale"
            ]
        )

        scaler.var_ = (
            scaler.scale_ ** 2
        )

        scaler.n_features_in_ = (
            len(
                scaler.mean_
            )
        )

        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        pca = PCA(
            n_components=N_QUBITS
        )

        pca.components_ = (
            preprocessor_data[
                "pca_components"
            ]
        )

        pca.mean_ = (
            preprocessor_data[
                "pca_mean"
            ]
        )

        # IMPORTANT:
        # These attributes are required by sklearn
        # for a properly reconstructed fitted PCA object.

        pca.n_features_in_ = (
            len(
                pca.mean_
            )
        )

        pca.n_components_ = (
            pca.components_.shape[0]
        )

        # ----------------------------------------------------
        # RESTORE PCA STATISTICS IF AVAILABLE
        # ----------------------------------------------------

        if (
            "pca_explained_variance"
            in preprocessor_data.files
        ):

            pca.explained_variance_ = (
                preprocessor_data[
                    "pca_explained_variance"
                ]
            )

        if (
            "pca_explained_variance_ratio"
            in preprocessor_data.files
        ):

            pca.explained_variance_ratio_ = (
                preprocessor_data[
                    "pca_explained_variance_ratio"
                ]
            )

        if (
            "pca_singular_values"
            in preprocessor_data.files
        ):

            pca.singular_values_ = (
                preprocessor_data[
                    "pca_singular_values"
                ]
            )

        # ----------------------------------------------------
        # ANGLE SCALER
        # ----------------------------------------------------

        angle_scaler = MinMaxScaler(
            feature_range=(0, np.pi)
        )

        angle_scaler.min_ = (
            preprocessor_data[
                "angle_min"
            ]
        )

        angle_scaler.scale_ = (
            preprocessor_data[
                "angle_scale"
            ]
        )

        angle_scaler.data_min_ = (
            preprocessor_data[
                "angle_data_min"
            ]
        )

        angle_scaler.data_max_ = (
            preprocessor_data[
                "angle_data_max"
            ]
        )

        angle_scaler.data_range_ = (
            preprocessor_data[
                "angle_data_range"
            ]
        )

        angle_scaler.n_features_in_ = (
            len(
                angle_scaler.data_min_
            )
        )

        angle_scaler.n_samples_seen_ = 1

        # ----------------------------------------------------
        # RETURN MODEL
        # ----------------------------------------------------

        return (
            weights,
            bias,
            scaler,
            pca,
            angle_scaler
        )

    except Exception as exc:

        print(
            f"Error loading VQC model "
            f"{model_file}: {exc}"
        )

        return None


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(
    features,
    scaler,
    pca,
    angle_scaler
):

    X = np.asarray(
        features,
        dtype=float
    ).reshape(
        1,
        -1
    )

    # --------------------------------------------------------
    # STANDARDIZE
    # --------------------------------------------------------

    X_scaled = (
        scaler.transform(
            X
        )
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    X_pca = (
        pca.transform(
            X_scaled
        )
    )

    # --------------------------------------------------------
    # CONVERT PCA FEATURES TO QUANTUM ANGLES
    # --------------------------------------------------------

    X_angles = (
        angle_scaler.transform(
            X_pca
        )
    )

    return X_angles[0]


# ============================================================
# VQC PREDICTION
# ============================================================

def predict_vqc(
    features,
    model_data
):

    if model_data is None:

        raise RuntimeError(
            "Quantum VQC model is not available."
        )

    (
        weights,
        bias,
        scaler,
        pca,
        angle_scaler
    ) = model_data

    # --------------------------------------------------------
    # PREPROCESS INPUT
    # --------------------------------------------------------

    x = prepare_input(
        features,
        scaler,
        pca,
        angle_scaler
    )

    # --------------------------------------------------------
    # QUANTUM INFERENCE
    # --------------------------------------------------------

    raw_output = float(
        variational_classifier(
            weights,
            bias,
            x
        )
    )

    # --------------------------------------------------------
    # CONVERT [-1, 1] TO [0, 1]
    # --------------------------------------------------------

    probability = float(
        np.clip(
            (raw_output + 1) / 2,
            0,
            1
        )
    )

    # --------------------------------------------------------
    # CLASS PREDICTION
    # --------------------------------------------------------

    prediction = int(
        raw_output >= 0
    )

    return (
        prediction,
        probability
    )


# ============================================================
# INITIALIZE QUANTUM MODELS
# ============================================================

def initialize_quantum_models():
    heart_model = load_vqc(
        HEART_MODEL_FILE,
        HEART_PREPROCESSOR_FILE
    )

    breast_model = load_vqc(
        BREAST_MODEL_FILE,
        BREAST_PREPROCESSOR_FILE
    )

    # ========================================================
    # TRAIN HEART VQC IF MODEL IS MISSING
    # ========================================================

    if heart_model is None:

        print(
            "\nTraining Quantum Heart Disease VQC..."
        )

        dataset_candidates = [
            PROJECT_ROOT / "heart_disease_uci.csv",
            PROJECT_ROOT / "heart.csv",
            PROJECT_ROOT.parent / "heart_disease_uci.csv",
            PROJECT_ROOT.parent / "heart.csv"
        ]

        heart_path = None

        for candidate in dataset_candidates:

            if candidate.exists():

                heart_path = candidate
                break

        if heart_path is None:

            raise FileNotFoundError(
                "Heart dataset not found.\n"
                "Expected one of:\n"
                + "\n".join(
                    str(path)
                    for path in dataset_candidates
                )
            )

        print(
            f"Using Heart dataset: {heart_path}"
        )

        heart_df = pd.read_csv(
            heart_path
        )

        # ----------------------------------------------------
        # Support both target and num column names
        # ----------------------------------------------------

        if "target" in heart_df.columns:

            X_df = heart_df.drop(
                columns=["target"]
            )

            y = heart_df[
                "target"
            ].values

        elif "num" in heart_df.columns:

            X_df = heart_df.drop(
                columns=["num"]
            )

            y = heart_df[
                "num"
            ].values

            y = (
                np.asarray(y) > 0
            ).astype(int)

        else:

            raise ValueError(
                "Heart dataset must contain "
                "'target' or 'num' column.\n"
                f"Available columns: "
                f"{list(heart_df.columns)}"
            )

        # ----------------------------------------------------
        # Convert categorical columns
        # ----------------------------------------------------

        X_df = pd.get_dummies(
            X_df,
            drop_first=False
        )

        X_df = X_df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X_df = X_df.fillna(
            X_df.median(
                numeric_only=True
            )
        )

        X_df = X_df.fillna(0)

        X = X_df.values

        X = np.asarray(
            X,
            dtype=float
        )

        y = np.asarray(
            y,
            dtype=int
        )

        print(
            f"Heart dataset shape: "
            f"{X.shape}"
        )

        print(
            f"Heart target shape: "
            f"{y.shape}"
        )

        (
            weights,
            bias,
            scaler,
            pca,
            angle_scaler,
            _,
            _
        ) = train_vqc(
            X,
            y
        )

        save_vqc(
            HEART_MODEL_FILE,
            HEART_PREPROCESSOR_FILE,
            weights,
            bias,
            scaler,
            pca,
            angle_scaler
        )

        heart_model = load_vqc(
            HEART_MODEL_FILE,
            HEART_PREPROCESSOR_FILE
        )

        print(
            "Quantum Heart Disease VQC "
            "training completed."
        )

    # ========================================================
    # TRAIN BREAST CANCER VQC IF MODEL IS MISSING
    # ========================================================

    if breast_model is None:

        print(
            "\nTraining Quantum Breast Cancer VQC..."
        )

        data = load_breast_cancer()

        X = data.data
        y = data.target

        (
            weights,
            bias,
            scaler,
            pca,
            angle_scaler,
            _,
            _
        ) = train_vqc(
            X,
            y
        )

        save_vqc(
            BREAST_MODEL_FILE,
            BREAST_PREPROCESSOR_FILE,
            weights,
            bias,
            scaler,
            pca,
            angle_scaler
        )

        breast_model = load_vqc(
            BREAST_MODEL_FILE,
            BREAST_PREPROCESSOR_FILE
        )

        print(
            "Quantum Breast Cancer VQC "
            "training completed."
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print(
        "\nQuantum model initialization complete."
    )

    print(
        "Quantum Heart VQC:",
        "READY"
        if heart_model is not None
        else "NOT AVAILABLE"
    )

    print(
        "Quantum Breast Cancer VQC:",
        "READY"
        if breast_model is not None
        else "NOT AVAILABLE"
    )

    return (
        heart_model,
        breast_model
    )