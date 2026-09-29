"""
Hybrid Quantum-Classical preprocessing pipeline.

Supported datasets:
    - WDBC / Breast Cancer
    - UCI Heart Disease

Pipeline:

    Raw dataset
        ↓
    Target encoding
        ↓
    Train/Test Split
        ↓
    StandardScaler
        ↓
    PCA → 4 components
        ↓
    MinMaxScaler → [0, π]
        ↓
    PyTorch tensors

This module is independent of the classical preprocessing pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

N_QUBITS = 4
TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# DATASET PATHS
# ============================================================

HEART_DATASET = (
    PROJECT_ROOT / "heart_disease_uci.csv"
)

BREAST_DATASET = (
    PROJECT_ROOT / "data.csv"
)


# ============================================================
# DATASET LOADING
# ============================================================

def load_hybrid_dataset(dataset):
    """
    Load a dataset for the hybrid model.

    Supported datasets:
        - heart
        - wdbc
        - breast
        - breast_cancer
    """

    dataset = dataset.lower().strip()

    # ========================================================
    # HEART DISEASE
    # ========================================================

    if dataset == "heart":

        if not HEART_DATASET.exists():

            raise FileNotFoundError(
                f"Heart Disease dataset not found:\n"
                f"{HEART_DATASET}"
            )

        df = pd.read_csv(
            HEART_DATASET
        )

        # ----------------------------------------------------
        # Find Heart Disease target column
        # ----------------------------------------------------

        if "target" in df.columns:

            X = df.drop(
                columns=["target"]
            )

            y = df["target"]

            target_column = "target"

        elif "num" in df.columns:

            X = df.drop(
                columns=["num"]
            )

            y = df["num"]

            target_column = "num"

            # ------------------------------------------------
            # Original UCI target:
            #
            # 0     = No Disease
            # 1-4   = Disease
            #
            # Convert to:
            #
            # 0 = No Disease
            # 1 = Disease
            # ------------------------------------------------

            y = pd.to_numeric(
                y,
                errors="coerce"
            )

            if y.isna().any():

                invalid_values = (
                    df.loc[
                        y.isna(),
                        target_column
                    ]
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )

                raise ValueError(
                    "Heart Disease target contains "
                    "invalid values.\n"
                    f"Actual values: {invalid_values}"
                )

            y = (
                y > 0
            ).astype(int)

        else:

            raise ValueError(
                "Heart Disease dataset must contain "
                "either a 'target' or 'num' column.\n"
                f"Available columns: {list(df.columns)}"
            )

    # ========================================================
    # BREAST CANCER / WDBC
    # ========================================================

    elif dataset in (
        "wdbc",
        "breast",
        "breast_cancer"
    ):

        if not BREAST_DATASET.exists():

            raise FileNotFoundError(
                f"Breast Cancer dataset not found:\n"
                f"{BREAST_DATASET}"
            )

        df = pd.read_csv(
            BREAST_DATASET
        )

        # ----------------------------------------------------
        # Find target column
        # ----------------------------------------------------

        target_column = None

        possible_targets = [
            "diagnosis",
            "Diagnosis",
            "target",
            "Target",
            "label",
            "Label",
            "class",
            "Class",
        ]

        for column in possible_targets:

            if column in df.columns:

                target_column = column
                break

        # ----------------------------------------------------
        # If normal target names are not found, inspect
        # columns for a likely two-class target.
        # ----------------------------------------------------

        if target_column is None:

            for column in df.columns:

                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .unique()
                )

                if len(values) == 2:

                    possible_values = set(
                        values
                    )

                    if possible_values.issubset(
                        {
                            "M",
                            "B",
                            "MALIGNANT",
                            "BENIGN",
                        }
                    ):

                        target_column = column
                        break

        if target_column is None:

            raise ValueError(
                "Could not find the breast cancer "
                "target column.\n"
                f"Available columns: {list(df.columns)}"
            )

        # ----------------------------------------------------
        # Separate features and target
        # ----------------------------------------------------

        X = df.drop(
            columns=[target_column]
        )

        y_original = df[
            target_column
        ].copy()

        # Keep original values for diagnostics.
        original_values = (
            y_original
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        print()
        print(
            f"WDBC target column: "
            f"{target_column}"
        )

        print(
            f"WDBC original target values: "
            f"{original_values}"
        )

        # ----------------------------------------------------
        # TARGET ENCODING
        # ----------------------------------------------------

        # Always normalize string representation first.
        y_string = (
            y_original
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # ----------------------------------------------------
        # Standard WDBC labels
        #
        # M = Malignant = 1
        # B = Benign    = 0
        # ----------------------------------------------------

        string_mapping = {
            "M": 1,
            "B": 0,
            "MALIGNANT": 1,
            "BENIGN": 0,
        }

        mapped_y = y_string.map(
            string_mapping
        )

        # ----------------------------------------------------
        # If every value was successfully mapped, use it.
        # ----------------------------------------------------

        if mapped_y.notna().all():

            y = mapped_y.astype(int)

        else:

            # ------------------------------------------------
            # Numeric / alternate encoding
            # ------------------------------------------------

            y_numeric = pd.to_numeric(
                y_original,
                errors="coerce"
            )

            # ----------------------------------------------
            # If numeric conversion works completely
            # ----------------------------------------------

            if y_numeric.notna().all():

                unique_values = sorted(
                    y_numeric.unique().tolist()
                )

                # Already 0/1
                if set(
                    unique_values
                ).issubset({0, 1}):

                    y = y_numeric.astype(int)

                # Exactly two numeric classes
                elif len(
                    unique_values
                ) == 2:

                    numeric_mapping = {
                        unique_values[0]: 0,
                        unique_values[1]: 1,
                    }

                    y = (
                        y_numeric
                        .map(numeric_mapping)
                        .astype(int)
                    )

                else:

                    raise ValueError(
                        "Breast Cancer target must "
                        "contain exactly two classes.\n"
                        f"Found values: {unique_values}"
                    )

            else:

                # ------------------------------------------
                # Some values could not be interpreted.
                # Show the exact values.
                # ------------------------------------------

                invalid_values = (
                    y_original[
                        y_numeric.isna()
                    ]
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )

                raise ValueError(
                    "Breast Cancer target contains "
                    "unrecognized values.\n"
                    f"Target column: {target_column}\n"
                    f"All target values: {original_values}\n"
                    f"Invalid values: {invalid_values}"
                )

    # ========================================================
    # UNSUPPORTED DATASET
    # ========================================================

    else:

        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            "Use 'heart' or 'wdbc'."
        )

    # ========================================================
    # FEATURE PREPROCESSING
    # ========================================================

    # Convert categorical feature columns into
    # numerical columns.
    X = pd.get_dummies(
        X,
        drop_first=True
    )

    # --------------------------------------------------------
    # Convert every feature to numeric
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove columns that are completely empty
    # --------------------------------------------------------

    X = X.dropna(
        axis=1,
        how="all"
    )

    # --------------------------------------------------------
    # Fill missing values using training-independent
    # column medians.
    # --------------------------------------------------------

    X = X.fillna(
        X.median(numeric_only=True)
    )

    # Final safety replacement.
    X = X.fillna(0)

    # ========================================================
    # TARGET VALIDATION
    # ========================================================

    # IMPORTANT:
    #
    # At this point y has already been converted to integer
    # 0/1 labels.
    #
    # Do NOT convert it again with pd.to_numeric(errors="coerce")
    # because that can destroy an already encoded target.

    y = np.asarray(
        y,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Check for invalid values
    # --------------------------------------------------------

    if np.isnan(
        y.astype(float)
    ).any():

        raise ValueError(
            f"{dataset} contains invalid target values."
        )

    # --------------------------------------------------------
    # Convert to pandas Series
    # --------------------------------------------------------

    y = pd.Series(
        y,
        index=X.index
    ).astype(int)

    # ========================================================
    # BINARY TARGET VALIDATION
    # ========================================================

    unique_labels = set(
        np.unique(y)
    )

    if not unique_labels.issubset(
        {0, 1}
    ):

        raise ValueError(
            f"{dataset} target must contain "
            f"only binary labels 0 and 1.\n"
            f"Found: {unique_labels}"
        )

    if len(unique_labels) != 2:

        raise ValueError(
            f"{dataset} must contain both "
            f"binary classes 0 and 1.\n"
            f"Found: {unique_labels}"
        )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print()

    print(
        f"Hybrid dataset loaded: "
        f"{dataset.upper()}"
    )

    print(
        f"Samples: {len(df)}"
    )

    print(
        f"Features after preprocessing: "
        f"{X.shape[1]}"
    )

    print(
        "Target distribution: "
        f"{dict(y.value_counts().sort_index())}"
    )

    print(
        f"Target column: "
        f"{target_column}"
    )

    return X, y


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_hybrid(dataset="wdbc"):
    """
    Prepare data for the hybrid quantum-classical model.

    Pipeline:

        Raw features
            ↓
        Train/Test Split
            ↓
        StandardScaler
            ↓
        PCA → 4 components
            ↓
        MinMaxScaler → [0, π]
            ↓
        PyTorch tensors

    Returns
    -------

    X_train : torch.Tensor
        Shape (N, 4).

    X_test : torch.Tensor
        Shape (N, 4).

    y_train : torch.Tensor
        Binary labels.

    y_test : torch.Tensor
        Binary labels.

    preprocessing_info : dict
        Information required for later inference.
    """

    dataset = dataset.lower().strip()

    # ========================================================
    # LOAD DATA
    # ========================================================

    X, y = load_hybrid_dataset(
        dataset
    )

    print()

    print("=" * 60)

    print(
        f"HYBRID PREPROCESSING — "
        f"{dataset.upper()}"
    )

    print("=" * 60)

    print(
        f"Dataset rows : {len(X)}"
    )

    print(
        f"Original features : {X.shape[1]}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        y.value_counts()
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        "\nTrain/Test Split:"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "y_test :",
        y_test.shape
    )

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ========================================================
    # PCA → 4 FEATURES
    # ========================================================

    if X_train_scaled.shape[1] < N_QUBITS:

        raise ValueError(
            f"Dataset has only "
            f"{X_train_scaled.shape[1]} features. "
            f"At least {N_QUBITS} are required."
        )

    pca = PCA(
        n_components=N_QUBITS
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    explained_variance = (
        pca.explained_variance_ratio_.sum()
    )

    print(
        f"\nPCA components : "
        f"{N_QUBITS}"
    )

    print(
        f"Explained variance : "
        f"{explained_variance:.4f}"
    )

    # ========================================================
    # ANGLE SCALING
    # ========================================================

    angle_scaler = MinMaxScaler(
        feature_range=(
            0,
            np.pi
        )
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

    print(
        f"\nTraining angle range : "
        f"[{X_train_angles.min():.4f}, "
        f"{X_train_angles.max():.4f}]"
    )

    print(
        f"Test angle range : "
        f"[{X_test_angles.min():.4f}, "
        f"{X_test_angles.max():.4f}]"
    )

    # ========================================================
    # CONVERT TO PYTORCH
    # ========================================================

    X_train_tensor = torch.tensor(
        X_train_angles,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test_angles,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test.to_numpy(),
        dtype=torch.float32
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    assert (
        X_train_tensor.shape[1]
        == N_QUBITS
    )

    assert (
        X_test_tensor.shape[1]
        == N_QUBITS
    )

    assert (
        X_train_tensor.dtype
        == torch.float32
    )

    assert (
        X_test_tensor.dtype
        == torch.float32
    )

    assert (
        y_train_tensor.dtype
        == torch.float32
    )

    assert (
        y_test_tensor.dtype
        == torch.float32
    )

    # ========================================================
    # QUANTUM-READY DATA SUMMARY
    # ========================================================

    print(
        "\nQuantum-ready hybrid data:"
    )

    print(
        "X_train:",
        X_train_tensor.shape
    )

    print(
        "X_test :",
        X_test_tensor.shape
    )

    print(
        "y_train:",
        y_train_tensor.shape
    )

    print(
        "y_test :",
        y_test_tensor.shape
    )

    # ========================================================
    # PREPROCESSING INFORMATION
    # ========================================================

    preprocessing_info = {

        "dataset":
            dataset,

        "n_qubits":
            N_QUBITS,

        "test_size":
            TEST_SIZE,

        "random_state":
            RANDOM_STATE,

        "dataset_rows":
            len(X),

        "original_features":
            X.shape[1],

        "pca_components":
            N_QUBITS,

        "explained_variance":
            float(
                explained_variance
            ),

        "train_rows":
            len(X_train),

        "test_rows":
            len(X_test),

        "class_distribution": {
            str(k): int(v)
            for k, v
            in y.value_counts().items()
        },

        "scaler":
            scaler,

        "pca":
            pca,

        "angle_scaler":
            angle_scaler,
    }

    return (
        X_train_tensor,
        X_test_tensor,
        y_train_tensor,
        y_test_tensor,
        preprocessing_info,
    )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "\nTesting hybrid WDBC preprocessing..."
    )

    preprocess_hybrid(
        "wdbc"
    )

    print(
        "\n\nTesting hybrid Heart Disease preprocessing..."
    )

    preprocess_hybrid(
        "heart"
    )