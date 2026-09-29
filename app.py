from flask import Flask, jsonify, request
from flask_cors import CORS
from model_service import (
    predict_heart,
    predict_breast_cancer,
    predict_quantum_heart,
    predict_quantum_breast_cancer,
    predict_hybrid_heart,
    predict_hybrid_breast_cancer,
    get_model_status,
    get_evaluation_results,
    get_benchmark_results,
)


app = Flask(__name__)

CORS(app)


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    status = get_model_status()

    return jsonify({
        "status": "healthy",

        "classical_models": {
            "heart": status.get("heart", False),
            "breast_cancer": status.get(
                "breast_cancer",
                False
            ),
        },

        "quantum_models": {
            "heart": status.get(
                "quantum_heart",
                False
            ),
            "breast_cancer": status.get(
                "quantum_breast_cancer",
                False
            ),
        },

        "hybrid_models": {
            "heart": status.get(
                "hybrid_heart",
                False
            ),
            "breast_cancer": status.get(
                "hybrid_breast_cancer",
                False
            ),
        }
    })


# ============================================================
# DATASETS
# ============================================================

@app.route("/api/datasets", methods=["GET"])
def datasets():

    return jsonify({
        "datasets": [

            {
                "id": "heart_disease",
                "name": "UCI Heart Disease",
                "features": 13,
                "classes": [
                    "No Disease",
                    "Disease"
                ]
            },

            {
                "id": "breast_cancer",
                "name": "Wisconsin Breast Cancer",
                "features": 30,
                "classes": [
                    "Benign",
                    "Malignant"
                ]
            }

        ]
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/api/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # READ REQUEST
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "success": False,
                "message": (
                    "Request body must contain JSON."
                )
            }), 400


        disease = data.get("disease")
        model = data.get("model")
        features = data.get("features")


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not disease:

            return jsonify({
                "success": False,
                "message": "Missing disease."
            }), 400


        if not model:

            return jsonify({
                "success": False,
                "message": "Missing model."
            }), 400


        if not isinstance(features, dict):

            return jsonify({
                "success": False,
                "message": (
                    "Features must be a JSON object."
                )
            }), 400


        # ====================================================
        # BREAST CANCER
        # ====================================================

        if disease == "breast_cancer":


            # ------------------------------------------------
            # CLASSICAL SVM
            # ------------------------------------------------

            if model == "svm":

                result = predict_breast_cancer(
                    features
                )

                return jsonify({

                    "success": True,

                    "disease": "breast_cancer",

                    "model": result[
                        "model_name"
                    ],

                    "prediction": result[
                        "prediction"
                    ],

                    "label": result[
                        "label"
                    ],

                    "probability": result[
                        "probability"
                    ],

                    "confidence": result[
                        "probability"
                    ],

                    "model_type": "classical",

                    "quantum_available": True,

                    "hybrid_available": True
                })


            # ------------------------------------------------
            # QUANTUM VQC
            # ------------------------------------------------

            if model == "quantum_vqc":

                result = (
                    predict_quantum_breast_cancer(
                        features
                    )
                )

                return jsonify({

                    "success": True,

                    "disease": "breast_cancer",

                    "model": result[
                        "model_name"
                    ],

                    "prediction": result[
                        "prediction"
                    ],

                    "label": result[
                        "label"
                    ],

                    "probability": result[
                        "probability"
                    ],

                    "confidence": result[
                        "probability"
                    ],

                    "model_type": "quantum",

                    "n_qubits": result.get(
                        "n_qubits",
                        4
                    ),

                    "n_layers": result.get(
                        "n_layers",
                        3
                    ),

                    "quantum_available": True,

                    "hybrid_available": True
                })


            # ------------------------------------------------
            # HYBRID VQC
            # ------------------------------------------------

            if model == "hybrid_vqc":

                result = (
                    predict_hybrid_breast_cancer(
                        features
                    )
                )

                return jsonify({

                    "success": True,

                    "disease": "breast_cancer",

                    "model": result[
                        "model_name"
                    ],

                    "prediction": result[
                        "prediction"
                    ],

                    "label": result[
                        "label"
                    ],

                    "probability": result[
                        "probability"
                    ],

                    "confidence": result[
                        "probability"
                    ],

                    "model_type": "hybrid",

                    "n_qubits": result.get(
                        "n_qubits",
                        4
                    ),

                    "n_layers": result.get(
                        "n_layers",
                        3
                    ),

                    "quantum_available": True,

                    "hybrid_available": True
                })


            # ------------------------------------------------
            # OTHER MODELS
            # ------------------------------------------------

            if model == "logistic_regression":

                return jsonify({
                    "success": False,
                    "message": (
                        "Logistic Regression "
                        "is not connected yet."
                    )
                }), 501


            if model == "random_forest":

                return jsonify({
                    "success": False,
                    "message": (
                        "Random Forest "
                        "is not connected yet."
                    )
                }), 501


            if model == "xgboost":

                return jsonify({
                    "success": False,
                    "message": (
                        "XGBoost "
                        "is not connected yet."
                    )
                }), 501


        # ====================================================
        # HEART DISEASE
        # ====================================================

        elif disease == "heart_disease":


            # ------------------------------------------------
            # CLASSICAL SVM
            # ------------------------------------------------

            if model == "svm":

                result = predict_heart(
                    features
                )

                return jsonify({

                    "success": True,

                    "disease": "heart_disease",

                    "model": result[
                        "model_name"
                    ],

                    "prediction": result[
                        "prediction"
                    ],

                    "label": result[
                        "label"
                    ],

                    "probability": result[
                        "probability"
                    ],

                    "confidence": result[
                        "probability"
                    ],

                    "model_type": "classical",

                    "quantum_available": True,

                    "hybrid_available": True
                })


            # ------------------------------------------------
            # QUANTUM VQC
            # ------------------------------------------------

            if model == "quantum_vqc":

                result = (
                    predict_quantum_heart(
                        features
                    )
                )

                return jsonify({

                    "success": True,

                    "disease": "heart_disease",

                    "model": result[
                        "model_name"
                    ],

                    "prediction": result[
                        "prediction"
                    ],

                    "label": result[
                        "label"
                    ],

                    "probability": result[
                        "probability"
                    ],

                    "confidence": result[
                        "probability"
                    ],

                    "model_type": "quantum",

                    "n_qubits": result.get(
                        "n_qubits",
                        4
                    ),

                    "n_layers": result.get(
                        "n_layers",
                        3
                    ),

                    "quantum_available": True,

                    "hybrid_available": True
                })


            # ------------------------------------------------
            # HYBRID VQC
            # ------------------------------------------------

            if model == "hybrid_vqc":

                result = (
                    predict_hybrid_heart(
                        features
                    )
                )

                return jsonify({

                    "success": True,

                    "disease": "heart_disease",

                    "model": result[
                        "model_name"
                    ],

                    "prediction": result[
                        "prediction"
                    ],

                    "label": result[
                        "label"
                    ],

                    "probability": result[
                        "probability"
                    ],

                    "confidence": result[
                        "probability"
                    ],

                    "model_type": "hybrid",

                    "n_qubits": result.get(
                        "n_qubits",
                        4
                    ),

                    "n_layers": result.get(
                        "n_layers",
                        3
                    ),

                    "quantum_available": True,

                    "hybrid_available": True
                })


            # ------------------------------------------------
            # OTHER MODELS
            # ------------------------------------------------

            if model == "logistic_regression":

                return jsonify({
                    "success": False,
                    "message": (
                        "Logistic Regression "
                        "is not connected yet."
                    )
                }), 501


            if model == "random_forest":

                return jsonify({
                    "success": False,
                    "message": (
                        "Random Forest "
                        "is not connected yet."
                    )
                }), 501


            if model == "xgboost":

                return jsonify({
                    "success": False,
                    "message": (
                        "XGBoost "
                        "is not connected yet."
                    )
                }), 501


        # ====================================================
        # UNSUPPORTED DISEASE
        # ====================================================

        else:

            return jsonify({

                "success": False,

                "message": (
                    f"Unsupported disease: "
                    f"{disease}"
                )

            }), 400


        # ====================================================
        # UNSUPPORTED MODEL
        # ====================================================

        return jsonify({

            "success": False,

            "message": (
                f"Unsupported model: {model}"
            )

        }), 400


    # ========================================================
    # VALUE ERROR
    # ========================================================

    except ValueError as exc:

        return jsonify({

            "success": False,

            "message": str(exc)

        }), 400


    # ========================================================
    # GENERAL ERROR
    # ========================================================

    except Exception as exc:

        print(
            "Prediction error:",
            exc
        )

        return jsonify({

            "success": False,

            "message": str(exc)

        }), 500


# ============================================================
# EVALUATION
# ============================================================

# ============================================================
# EVALUATION
# ============================================================

@app.route("/api/evaluation", methods=["GET"])
def evaluation():

    try:

        results = get_evaluation_results()

        return jsonify(results)

    except Exception as exc:

        print(
            "Evaluation error:",
            exc
        )

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# BENCHMARKS
# ============================================================

# ============================================================
# BENCHMARKS
# ============================================================

@app.route("/api/benchmarks", methods=["GET"])
def benchmarks():

    try:

        results = get_benchmark_results()

        return jsonify(results)

    except Exception as exc:

        print(
            "Benchmark error:",
            exc
        )

        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )