from pathlib import Path
import json
import pickle
import joblib

import pandas as pd
import numpy as np
import torch

from hybrid.hybrid_model import HybridQuantumClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "Results"


# ============================================================
# CLASSICAL MODEL PATHS
# ============================================================

MODEL_REPO = (
    PROJECT_ROOT.parent
    / "Disease-detector-classical-model"
)


HEART_MODEL_PATH = (
    MODEL_REPO
    / "heart_disease"
    / "heart_disease_classical_model.joblib"
)


BREAST_MODEL_PATH = (
    MODEL_REPO
    / "breast_cancer"
    / "breast_cancer_classical_model.joblib"
)


# ============================================================
# DATASET PATHS
# ============================================================

HEART_DATASET_PATH = (
    PROJECT_ROOT
    / "heart_disease_uci.csv"
)


BREAST_DATASET_PATH = (
    PROJECT_ROOT
    / "data.csv"
)


# ============================================================
# HYBRID MODEL PATHS
# ============================================================

HYBRID_HEART_MODEL_PATH = (
    RESULTS_DIR
    / "hybrid_heart_model.pt"
)


HYBRID_HEART_PREPROCESSOR_PATH = (
    RESULTS_DIR
    / "hybrid_heart_preprocessor.pkl"
)


HYBRID_WDBC_MODEL_PATH = (
    RESULTS_DIR
    / "hybrid_wdbc_model.pt"
)


HYBRID_WDBC_PREPROCESSOR_PATH = (
    RESULTS_DIR
    / "hybrid_wdbc_preprocessor.pkl"
)


# ============================================================
# LOAD CLASSICAL MODELS
# ============================================================

def load_models():

    if not HEART_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Heart model not found:\n"
            f"{HEART_MODEL_PATH}"
        )

    if not BREAST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Breast cancer model not found:\n"
            f"{BREAST_MODEL_PATH}"
        )

    heart_model = joblib.load(
        HEART_MODEL_PATH
    )

    breast_model = joblib.load(
        BREAST_MODEL_PATH
    )

    return (
        heart_model,
        breast_model
    )


heart_model, breast_model = load_models()


# ============================================================
# LOAD QUANTUM VQC MODELS
# ============================================================

print("\nLoading Quantum VQC models...")


quantum_heart_model = None
quantum_breast_model = None


try:

    from quantum_inference import (
        load_vqc,
        predict_vqc,
        HEART_MODEL_FILE,
        HEART_PREPROCESSOR_FILE,
        BREAST_MODEL_FILE,
        BREAST_PREPROCESSOR_FILE,
    )

    # --------------------------------------------------------
    # HEART VQC
    # --------------------------------------------------------

    if (
        HEART_MODEL_FILE.exists()
        and HEART_PREPROCESSOR_FILE.exists()
    ):

        try:

            quantum_heart_model = load_vqc(
                HEART_MODEL_FILE,
                HEART_PREPROCESSOR_FILE
            )

        except Exception as exc:

            print(
                "Quantum Heart VQC loading warning:",
                exc
            )

            quantum_heart_model = None

    else:

        print(
            "Quantum Heart VQC files not found."
        )

    # --------------------------------------------------------
    # BREAST CANCER VQC
    # --------------------------------------------------------

    if (
        BREAST_MODEL_FILE.exists()
        and BREAST_PREPROCESSOR_FILE.exists()
    ):

        try:

            quantum_breast_model = load_vqc(
                BREAST_MODEL_FILE,
                BREAST_PREPROCESSOR_FILE
            )

        except Exception as exc:

            print(
                "Quantum Breast Cancer VQC "
                "loading warning:",
                exc
            )

            quantum_breast_model = None

    else:

        print(
            "Quantum Breast Cancer VQC "
            "files not found."
        )


except Exception as exc:

    print(
        "Quantum model loading warning:",
        exc
    )


print(
    "Quantum Heart VQC:",
    "READY"
    if quantum_heart_model is not None
    else "NOT AVAILABLE"
)


print(
    "Quantum Breast Cancer VQC:",
    "READY"
    if quantum_breast_model is not None
    else "NOT AVAILABLE"
)


# ============================================================
# LOAD HYBRID MODELS
# ============================================================

def load_hybrid_model(
    model_path
):

    if not model_path.exists():
        return None

    model = HybridQuantumClassifier()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


def load_hybrid_preprocessor(
    preprocessor_path
):

    if not preprocessor_path.exists():
        return None

    with open(
        preprocessor_path,
        "rb"
    ) as file:

        return pickle.load(file)


hybrid_heart_model = (
    load_hybrid_model(
        HYBRID_HEART_MODEL_PATH
    )
)


hybrid_wdbc_model = (
    load_hybrid_model(
        HYBRID_WDBC_MODEL_PATH
    )
)


hybrid_heart_preprocessor = (
    load_hybrid_preprocessor(
        HYBRID_HEART_PREPROCESSOR_PATH
    )
)


hybrid_wdbc_preprocessor = (
    load_hybrid_preprocessor(
        HYBRID_WDBC_PREPROCESSOR_PATH
    )
)


# ============================================================
# HEART FEATURES
# ============================================================

HEART_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


HEART_CATEGORICAL_COLUMNS = [
    "sex",
    "cp",
    "restecg",
    "slope",
    "thal",
]


# ============================================================
# BREAST CANCER FEATURES
# ============================================================

BREAST_FEATURES = [

    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",

]


# ============================================================
# HEART CLASSICAL INPUT PREPARATION
# ============================================================

def prepare_heart_features(
    features
):

    for feature in HEART_FEATURES:

        if feature not in features:

            raise ValueError(
                f"Missing required feature: "
                f"{feature}"
            )

    data = {

        "age":
            float(features["age"]),

        "trestbps":
            float(features["trestbps"]),

        "chol":
            float(features["chol"]),

        "fbs":
            int(features["fbs"]),

        "thalch":
            float(features["thalch"]),

        "exang":
            int(features["exang"]),

        "oldpeak":
            float(features["oldpeak"]),

        "ca":
            float(features["ca"]),

        "sex":
            features["sex"],

        "cp":
            features["cp"],

        "restecg":
            features["restecg"],

        "slope":
            features["slope"],

        "thal":
            features["thal"],
    }

    df = pd.DataFrame(
        [data]
    )

    for column in HEART_CATEGORICAL_COLUMNS:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    df["sex"] = df["sex"].replace({

        "female": "Female",
        "male": "Male",

    })

    df = pd.get_dummies(
        df,
        columns=HEART_CATEGORICAL_COLUMNS,
        drop_first=False
    )

    bool_columns = (
        df.select_dtypes(
            include="bool"
        ).columns
    )

    df[bool_columns] = (
        df[bool_columns]
        .astype(int)
    )

    expected_columns = list(
        heart_model.feature_names_in_
    )

    for column in expected_columns:

        if column not in df.columns:

            df[column] = 0

    df = df[
        expected_columns
    ]

    return df


# ============================================================
# CLASSICAL HEART PREDICTION
# ============================================================

def predict_heart(
    features
):

    df = prepare_heart_features(
        features
    )

    prediction = int(
        heart_model.predict(df)[0]
    )

    probabilities = (
        heart_model.predict_proba(df)[0]
    )

    class_to_probability = {

        int(cls): float(prob)

        for cls, prob in zip(
            heart_model.classes_,
            probabilities
        )

    }

    probability = (
        class_to_probability[
            prediction
        ]
    )

    return {

        "prediction":
            prediction,

        "label":
            (
                "Disease"
                if prediction == 1
                else "No Disease"
            ),

        "probability":
            probability,

        "model_name":
            "Classical Heart Disease SVM",

        "model_type":
            "classical",

    }


# ============================================================
# CLASSICAL BREAST CANCER INPUT
# ============================================================

def prepare_breast_features(
    features
):

    values = {}

    for feature in BREAST_FEATURES:

        if feature not in features:

            raise ValueError(
                f"Missing required feature: "
                f"{feature}"
            )

        try:

            values[feature] = float(
                features[feature]
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                f"Feature '{feature}' "
                f"must be numeric."
            )

    df = pd.DataFrame(
        [values]
    )

    return df[
        BREAST_FEATURES
    ]


# ============================================================
# CLASSICAL BREAST CANCER PREDICTION
# ============================================================

def predict_breast_cancer(
    features
):

    df = prepare_breast_features(
        features
    )

    prediction = int(
        breast_model.predict(df)[0]
    )

    probabilities = (
        breast_model.predict_proba(df)[0]
    )

    class_to_probability = {

        int(cls): float(prob)

        for cls, prob in zip(
            breast_model.classes_,
            probabilities
        )

    }

    probability = (
        class_to_probability[
            prediction
        ]
    )

    return {

        "prediction":
            prediction,

        "label":
            (
                "Malignant"
                if prediction == 1
                else "Benign"
            ),

        "probability":
            probability,

        "model_name":
            "Classical Breast Cancer SVM",

        "model_type":
            "classical",

    }


# ============================================================
# QUANTUM HEART FEATURE PREPARATION
# ============================================================

def prepare_quantum_heart_features(
    features
):

    values = []

    for feature in HEART_FEATURES:

        if feature not in features:

            raise ValueError(
                f"Missing required feature: "
                f"{feature}"
            )

        value = features[feature]

        # ----------------------------------------------------
        # Numeric features
        # ----------------------------------------------------

        if feature in [

            "age",
            "trestbps",
            "chol",
            "fbs",
            "thalch",
            "exang",
            "oldpeak",
            "ca",

        ]:

            try:

                values.append(
                    float(value)
                )

            except (
                TypeError,
                ValueError
            ):

                raise ValueError(
                    f"Feature '{feature}' "
                    f"must be numeric."
                )

        # ----------------------------------------------------
        # Sex
        # ----------------------------------------------------

        elif feature == "sex":

            value = str(
                value
            ).strip().lower()

            if value in [
                "male",
                "m",
                "1",
            ]:

                values.append(1.0)

            elif value in [
                "female",
                "f",
                "0",
            ]:

                values.append(0.0)

            else:

                raise ValueError(
                    "Invalid sex value. "
                    "Use Male or Female."
                )

        # ----------------------------------------------------
        # Chest Pain
        # ----------------------------------------------------

        elif feature == "cp":

            value = str(
                value
            ).strip().lower()

            cp_mapping = {

                "typical angina": 0,
                "atypical angina": 1,
                "non-anginal pain": 2,
                "non anginal pain": 2,
                "asymptomatic": 3,

            }

            if value in cp_mapping:

                values.append(
                    float(
                        cp_mapping[value]
                    )
                )

            else:

                try:

                    values.append(
                        float(value)
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    raise ValueError(
                        f"Invalid cp value: "
                        f"{value}"
                    )

        # ----------------------------------------------------
        # Rest ECG
        # ----------------------------------------------------

        elif feature == "restecg":

            value = str(
                value
            ).strip().lower()

            restecg_mapping = {

                "normal": 0,
                "st-t wave abnormality": 1,
                "st-t": 1,
                "left ventricular hypertrophy": 2,
                "lv hypertrophy": 2,

            }

            if value in restecg_mapping:

                values.append(
                    float(
                        restecg_mapping[value]
                    )
                )

            else:

                try:

                    values.append(
                        float(value)
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    raise ValueError(
                        f"Invalid restecg value: "
                        f"{value}"
                    )

        # ----------------------------------------------------
        # Slope
        # ----------------------------------------------------

        elif feature == "slope":

            value = str(
                value
            ).strip().lower()

            slope_mapping = {

                "upsloping": 0,
                "flat": 1,
                "downsloping": 2,

            }

            if value in slope_mapping:

                values.append(
                    float(
                        slope_mapping[value]
                    )
                )

            else:

                try:

                    values.append(
                        float(value)
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    raise ValueError(
                        f"Invalid slope value: "
                        f"{value}"
                    )

        # ----------------------------------------------------
        # Thal
        # ----------------------------------------------------

        elif feature == "thal":

            value = str(
                value
            ).strip().lower()

            thal_mapping = {

                "normal": 3,
                "fixed defect": 6,
                "fixed": 6,
                "reversible defect": 7,
                "reversible": 7,

            }

            if value in thal_mapping:

                values.append(
                    float(
                        thal_mapping[value]
                    )
                )

            else:

                try:

                    values.append(
                        float(value)
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    raise ValueError(
                        f"Invalid thal value: "
                        f"{value}"
                    )

    return values


# ============================================================
# QUANTUM HEART PREDICTION
# ============================================================

def predict_quantum_heart(
    features
):

    if quantum_heart_model is None:

        raise RuntimeError(
            "Quantum Heart Disease VQC "
            "is not available. "
            "Train/save the quantum model first."
        )

    values = (
        prepare_quantum_heart_features(
            features
        )
    )

    prediction, probability = (
        predict_vqc(
            values,
            quantum_heart_model
        )
    )

    return {

        "prediction":
            prediction,

        "label":
            (
                "Disease"
                if prediction == 1
                else "No Disease"
            ),

        "probability":
            probability,

        "model_name":
            "Quantum VQC "
            "(4 qubits) - Heart Disease",

        "model_type":
            "quantum",

        "n_qubits":
            4,

        "n_layers":
            3,

    }


# ============================================================
# QUANTUM BREAST CANCER PREDICTION
# ============================================================

def predict_quantum_breast_cancer(
    features
):

    if quantum_breast_model is None:

        raise RuntimeError(
            "Quantum Breast Cancer VQC "
            "is not available. "
            "Train/save the quantum model first."
        )

    values = []

    for feature in BREAST_FEATURES:

        if feature not in features:

            raise ValueError(
                f"Missing required feature: "
                f"{feature}"
            )

        try:

            values.append(
                float(
                    features[feature]
                )
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                f"Feature '{feature}' "
                f"must be numeric."
            )

    prediction, probability = (
        predict_vqc(
            values,
            quantum_breast_model
        )
    )

    return {

        "prediction":
            prediction,

        "label":
            (
                "Malignant"
                if prediction == 1
                else "Benign"
            ),

        "probability":
            probability,

        "model_name":
            "Quantum VQC "
            "(4 qubits) - Breast Cancer",

        "model_type":
            "quantum",

        "n_qubits":
            4,

        "n_layers":
            3,

    }


# ============================================================
# HYBRID FEATURE TRANSFORMATION
# ============================================================

def _transform_hybrid_features(
    df,
    preprocessor
):

    if preprocessor is None:

        raise RuntimeError(
            "Hybrid preprocessing pipeline "
            "is not available."
        )

    scaler = (
        preprocessor["scaler"]
    )

    pca = (
        preprocessor["pca"]
    )

    angle_scaler = (
        preprocessor["angle_scaler"]
    )

    expected_columns = list(
        scaler.feature_names_in_
    )

    for column in expected_columns:

        if column not in df.columns:

            df[column] = 0

    df = df[
        expected_columns
    ]

    scaled = scaler.transform(
        df
    )

    pca_features = pca.transform(
        scaled
    )

    angles = angle_scaler.transform(
        pca_features
    )

    return torch.tensor(
        angles,
        dtype=torch.float32
    )


# ============================================================
# HYBRID HEART INPUT
# ============================================================

def prepare_hybrid_heart_features(
    features
):

    if hybrid_heart_preprocessor is None:

        raise RuntimeError(
            "Hybrid Heart Disease "
            "preprocessing is not available."
        )

    df = prepare_heart_features(
        features
    )

    return _transform_hybrid_features(
        df,
        hybrid_heart_preprocessor
    )


# ============================================================
# HYBRID HEART PREDICTION
# ============================================================

def predict_hybrid_heart(
    features
):

    if hybrid_heart_model is None:

        raise RuntimeError(
            "Hybrid Heart Disease model "
            "is not available."
        )

    tensor = (
        prepare_hybrid_heart_features(
            features
        )
    )

    with torch.no_grad():

        logits = (
            hybrid_heart_model(
                tensor
            )
        )

        probability = float(
            torch.sigmoid(
                logits
            )[0].item()
        )

    prediction = int(
        probability >= 0.5
    )

    return {

        "prediction":
            prediction,

        "label":
            (
                "Disease"
                if prediction == 1
                else "No Disease"
            ),

        "probability":
            probability,

        "model_name":
            "Hybrid Quantum-Classical "
            "VQC Heart Disease",

        "model_type":
            "hybrid",

    }


# ============================================================
# HYBRID BREAST CANCER INPUT
# ============================================================

def prepare_hybrid_breast_features(
    features
):

    if hybrid_wdbc_preprocessor is None:

        raise RuntimeError(
            "Hybrid Breast Cancer "
            "preprocessing is not available."
        )

    df = prepare_breast_features(
        features
    )

    return _transform_hybrid_features(
        df,
        hybrid_wdbc_preprocessor
    )


# ============================================================
# HYBRID BREAST CANCER PREDICTION
# ============================================================

def predict_hybrid_breast_cancer(
    features
):

    if hybrid_wdbc_model is None:

        raise RuntimeError(
            "Hybrid Breast Cancer model "
            "is not available."
        )

    tensor = (
        prepare_hybrid_breast_features(
            features
        )
    )

    with torch.no_grad():

        logits = (
            hybrid_wdbc_model(
                tensor
            )
        )

        probability = float(
            torch.sigmoid(
                logits
            )[0].item()
        )

    prediction = int(
        probability >= 0.5
    )

    return {

        "prediction":
            prediction,

        "label":
            (
                "Malignant"
                if prediction == 1
                else "Benign"
            ),

        "probability":
            probability,

        "model_name":
            "Hybrid Quantum-Classical "
            "VQC Breast Cancer",

        "model_type":
            "hybrid",

    }


# ============================================================
# DATASET LOADING
# ============================================================

def load_heart_dataset():

    if not HEART_DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Heart dataset not found:\n"
            f"{HEART_DATASET_PATH}"
        )

    return pd.read_csv(
        HEART_DATASET_PATH
    )


def load_breast_dataset():

    if not BREAST_DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Breast cancer dataset not found:\n"
            f"{BREAST_DATASET_PATH}"
        )

    return pd.read_csv(
        BREAST_DATASET_PATH
    )


# ============================================================
# HEART DATASET → MODEL FORMAT
# ============================================================

def heart_dataset_to_model_data(
    df
):

    result_rows = []
    labels = []

    required = [

        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalch",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",

    ]

    for _, row in df.iterrows():

        if row[required].isna().any():

            continue

        features = {

            "age":
                row["age"],

            "sex":
                row["sex"],

            "cp":
                row["cp"],

            "trestbps":
                row["trestbps"],

            "chol":
                row["chol"],

            "fbs":
                row["fbs"],

            "restecg":
                row["restecg"],

            "thalch":
                row["thalch"],

            "exang":
                row["exang"],

            "oldpeak":
                row["oldpeak"],

            "slope":
                row["slope"],

            "ca":
                row["ca"],

            "thal":
                row["thal"],

        }

        try:

            prepared = (
                prepare_heart_features(
                    features
                )
            )

            result_rows.append(
                prepared.iloc[0]
            )

            labels.append(

                1
                if int(row["num"]) > 0
                else 0

            )

        except Exception:

            continue

    X = pd.DataFrame(
        result_rows
    )

    X = X.reindex(
        columns=
        heart_model.feature_names_in_,
        fill_value=0
    )

    y = np.array(
        labels
    )

    return X, y


# ============================================================
# BREAST DATASET → MODEL FORMAT
# ============================================================

def breast_dataset_to_model_data(
    df
):

    rename_map = {

        "concave points_mean":
            "concave_points_mean",

        "concave points_se":
            "concave_points_se",

        "concave points_worst":
            "concave_points_worst",

    }

    df = df.rename(
        columns=rename_map
    )

    X = df[
        BREAST_FEATURES
    ].copy()

    for column in BREAST_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    valid_rows = ~X.isna().any(
        axis=1
    )

    X = X.loc[
        valid_rows
    ]

    y = (
        df.loc[
            valid_rows,
            "diagnosis"
        ]
        .map({

            "B": 0,
            "M": 1,

        })
        .astype(int)
        .to_numpy()
    )

    return X, y


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability
):

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0

    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_true,
            y_probability
        )

    except ValueError:

        roc_auc = 0.0

    return {

        "accuracy":
            round(
                float(accuracy),
                4
            ),

        "precision":
            round(
                float(precision),
                4
            ),

        "sensitivity":
            round(
                float(sensitivity),
                4
            ),

        "specificity":
            round(
                float(specificity),
                4
            ),

        "f1_score":
            round(
                float(f1),
                4
            ),

        "roc_auc":
            round(
                float(roc_auc),
                4
            ),

        "confusion_matrix": {

            "true_negative":
                int(tn),

            "false_positive":
                int(fp),

            "false_negative":
                int(fn),

            "true_positive":
                int(tp),

        }

    }


# ============================================================
# CLASSICAL EVALUATION
# ============================================================

def evaluate_breast_cancer():

    dataset = (
        load_breast_dataset()
    )

    X, y_true = (
        breast_dataset_to_model_data(
            dataset
        )
    )

    y_pred = (
        breast_model.predict(X)
    )

    probabilities = (
        breast_model.predict_proba(X)
    )

    class_to_index = {

        int(cls): index

        for index, cls in enumerate(
            breast_model.classes_
        )

    }

    positive_index = (
        class_to_index.get(1)
    )

    if positive_index is not None:

        y_probability = (
            probabilities[
                :,
                positive_index
            ]
        )

    else:

        y_probability = (
            probabilities[:, -1]
        )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_probability
    )

    metrics.update({

        "model":
            "Classical Breast Cancer SVM",

        "dataset":
            "Wisconsin Breast Cancer",

        "dataset_rows":
            int(len(dataset)),

        "evaluated_rows":
            int(len(y_true)),

        "classes": {

            "0":
                "Benign",

            "1":
                "Malignant",

        }

    })

    return metrics


def evaluate_heart():

    dataset = (
        load_heart_dataset()
    )

    X, y_true = (
        heart_dataset_to_model_data(
            dataset
        )
    )

    y_pred = (
        heart_model.predict(X)
    )

    probabilities = (
        heart_model.predict_proba(X)
    )

    class_to_index = {

        int(cls): index

        for index, cls in enumerate(
            heart_model.classes_
        )

    }

    positive_index = (
        class_to_index.get(1)
    )

    if positive_index is not None:

        y_probability = (
            probabilities[
                :,
                positive_index
            ]
        )

    else:

        y_probability = (
            probabilities[:, -1]
        )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_probability
    )

    metrics.update({

        "model":
            "Classical Heart Disease SVM",

        "dataset":
            "UCI Heart Disease",

        "dataset_rows":
            int(len(dataset)),

        "evaluated_rows":
            int(len(y_true)),

        "classes": {

            "0":
                "No Disease",

            "1":
                "Disease",

        }

    })

    return metrics


# ============================================================
# HYBRID BENCHMARK RESULTS
# ============================================================

def load_hybrid_benchmark_results():

    results_file = (
        RESULTS_DIR
        / "hybrid_benchmark_results.json"
    )

    if not results_file.exists():

        return []

    try:

        with open(
            results_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        return data.get(
            "models",
            []
        )

    except (
        OSError,
        json.JSONDecodeError,
    ):

        return []


# ============================================================
# EVALUATION RESULTS
# ============================================================

def get_evaluation_results():

    heart = evaluate_heart()

    breast = (
        evaluate_breast_cancer()
    )

    hybrid_models = (
        load_hybrid_benchmark_results()
    )

    hybrid_heart = next(

        (
            model
            for model in hybrid_models

            if model.get(
                "disease"
            )
            == "Heart Disease"
        ),

        None

    )

    hybrid_breast = next(

        (
            model
            for model in hybrid_models

            if model.get(
                "disease"
            )
            == "Breast Cancer"
        ),

        None

    )

    quantum_file = (
        RESULTS_DIR
        / "quantum_benchmark_results.json"
    )

    quantum_results = {

        "heart_disease":
            None,

        "breast_cancer":
            None,

    }

    if quantum_file.exists():

        try:

            with open(
                quantum_file,
                "r",
                encoding="utf-8"
            ) as file:

                quantum_data = json.load(
                    file
                )

            quantum_models = (
                quantum_data.get(
                    "models",
                    []
                )
            )

            for model in quantum_models:

                if model.get(
                    "type"
                ) != "quantum":

                    continue

                disease = (
                    model.get(
                        "disease",
                        ""
                    )
                    .strip()
                    .lower()
                )

                if disease == "heart disease":

                    quantum_results[
                        "heart_disease"
                    ] = model

                elif disease == "breast cancer":

                    quantum_results[
                        "breast_cancer"
                    ] = model

        except (
            OSError,
            json.JSONDecodeError,
        ):

            pass

    return {

        "status":
            "success",

        "heart_disease":
            heart,

        "breast_cancer":
            breast,

        "quantum":
            quantum_results,

        "hybrid": {

            "heart_disease":
                hybrid_heart,

            "breast_cancer":
                hybrid_breast,

        }

    }


# ============================================================
# BENCHMARK RESULTS
# ============================================================

def get_benchmark_results():

    heart = evaluate_heart()

    breast = (
        evaluate_breast_cancer()
    )

    quantum_file = (
        RESULTS_DIR
        / "quantum_benchmark_results.json"
    )

    quantum_models = []

    if quantum_file.exists():

        try:

            with open(
                quantum_file,
                "r",
                encoding="utf-8"
            ) as file:

                quantum_data = json.load(
                    file
                )

            quantum_models = (
                quantum_data.get(
                    "models",
                    []
                )
            )

        except (
            OSError,
            json.JSONDecodeError,
        ):

            quantum_models = []

    models = [

        {
            "name":
                "Classical Heart Disease SVM",

            "disease":
                "Heart Disease",

            "type":
                "classical",

            "model":
                heart["model"],

            "dataset_rows":
                heart["dataset_rows"],

            "evaluated_rows":
                heart["evaluated_rows"],

            "accuracy":
                heart["accuracy"] * 100,

            "sensitivity":
                heart["sensitivity"] * 100,

            "specificity":
                heart["specificity"] * 100,

            "precision":
                heart["precision"] * 100,

            "f1":
                heart["f1_score"] * 100,

            "roc_auc":
                heart["roc_auc"] * 100,

        },

        {
            "name":
                "Classical Breast Cancer SVM",

            "disease":
                "Breast Cancer",

            "type":
                "classical",

            "model":
                breast["model"],

            "dataset_rows":
                breast["dataset_rows"],

            "evaluated_rows":
                breast["evaluated_rows"],

            "accuracy":
                breast["accuracy"] * 100,

            "sensitivity":
                breast["sensitivity"] * 100,

            "specificity":
                breast["specificity"] * 100,

            "precision":
                breast["precision"] * 100,

            "f1":
                breast["f1_score"] * 100,

            "roc_auc":
                breast["roc_auc"] * 100,

        },

    ]


    # --------------------------------------------------------
    # QUANTUM BENCHMARKS
    # --------------------------------------------------------

    for quantum in quantum_models:

        models.append({

            "name":
                quantum.get(
                    "name"
                ),

            "disease":
                quantum.get(
                    "disease"
                ),

            "type":
                "quantum",

            "model":
                quantum.get(
                    "model"
                ),

            "dataset":
                quantum.get(
                    "dataset"
                ),

            "dataset_rows":
                quantum.get(
                    "dataset_rows"
                ),

            "evaluated_rows":
                quantum.get(
                    "evaluated_rows"
                ),

            "accuracy":
                quantum.get(
                    "accuracy"
                ),

            "sensitivity":
                quantum.get(
                    "sensitivity"
                ),

            "specificity":
                quantum.get(
                    "specificity"
                ),

            "precision":
                quantum.get(
                    "precision"
                ),

            "f1":
                quantum.get(
                    "f1"
                ),

            "roc_auc":
                quantum.get(
                    "roc_auc"
                ),

            "confusion_matrix":
                quantum.get(
                    "confusion_matrix"
                ),

            "true_positive":
                quantum.get(
                    "true_positive"
                ),

            "true_negative":
                quantum.get(
                    "true_negative"
                ),

            "false_positive":
                quantum.get(
                    "false_positive"
                ),

            "false_negative":
                quantum.get(
                    "false_negative"
                ),

        })


    # --------------------------------------------------------
    # HYBRID BENCHMARKS
    # --------------------------------------------------------

    hybrid_models = (
        load_hybrid_benchmark_results()
    )

    for hybrid in hybrid_models:

        models.append({

            "name":
                hybrid.get(
                    "name"
                ),

            "disease":
                hybrid.get(
                    "disease"
                ),

            "type":
                "hybrid",

            "model":
                hybrid.get(
                    "model"
                ),

            "dataset":
                hybrid.get(
                    "dataset"
                ),

            "dataset_rows":
                hybrid.get(
                    "dataset_rows"
                ),

            "evaluated_rows":
                hybrid.get(
                    "evaluated_rows"
                ),

            "accuracy":
                hybrid.get(
                    "accuracy"
                ),

            "sensitivity":
                hybrid.get(
                    "sensitivity"
                ),

            "specificity":
                hybrid.get(
                    "specificity"
                ),

            "precision":
                hybrid.get(
                    "precision"
                ),

            "f1":
                hybrid.get(
                    "f1"
                ),

            "roc_auc":
                hybrid.get(
                    "roc_auc"
                ),

            "confusion_matrix":
                hybrid.get(
                    "confusion_matrix"
                ),

            "true_positive":
                hybrid.get(
                    "true_positive"
                ),

            "true_negative":
                hybrid.get(
                    "true_negative"
                ),

            "false_positive":
                hybrid.get(
                    "false_positive"
                ),

            "false_negative":
                hybrid.get(
                    "false_negative"
                ),

            "training_time_seconds":
                hybrid.get(
                    "training_time_seconds"
                ),

            "inference_time_seconds":
                hybrid.get(
                    "inference_time_seconds"
                ),

            "n_qubits":
                hybrid.get(
                    "n_qubits"
                ),

            "n_layers":
                hybrid.get(
                    "n_layers"
                ),

        })


    return {

        "status":
            "success",

        "models":
            models,

        "note":
            (
                "Benchmark results include "
                "classical SVM, quantum VQC, "
                "and hybrid quantum-classical "
                "VQC models."
            ),

    }


# ============================================================
# MODEL STATUS
# ============================================================

def get_model_status():

    return {

        "heart":
            heart_model is not None,

        "breast_cancer":
            breast_model is not None,

        "quantum_heart":
            quantum_heart_model is not None,

        "quantum_breast_cancer":
            quantum_breast_model is not None,

        "hybrid_heart":
            hybrid_heart_model is not None,

        "hybrid_breast_cancer":
            hybrid_wdbc_model is not None,

    }