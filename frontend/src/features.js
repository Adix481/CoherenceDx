// ============================================================
// BREAST CANCER FEATURES
// ============================================================

export const FEATURE_GROUPS = {
  mean: [
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
  ],

  se: [
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
  ],

  worst: [
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
  ],
};


// ============================================================
// BREAST CANCER FEATURE ORDER
// ============================================================

export const FEATURE_ORDER = [
  ...FEATURE_GROUPS.mean,
  ...FEATURE_GROUPS.se,
  ...FEATURE_GROUPS.worst,
];


// ============================================================
// HEART DISEASE FEATURES
//
// These are the 13 raw clinical features expected by
// the Flask backend.
// ============================================================

export const HEART_FEATURES = [
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
];


// ============================================================
// MODEL OPTIONS
//
// Currently connected models:
//
// 1. Hybrid VQC
// 2. SVM
//
// Other classical models are intentionally not exposed
// until their Flask inference endpoints are connected.
// ============================================================

export const MODEL_OPTIONS = [
  {
    id: "hybrid_vqc",
    label: "Hybrid VQC (4 qubits)",
  },

  {
    id: "svm",
    label: "SVM",
  },
];


// ============================================================
// BREAST CANCER LABEL MAP
// ============================================================

export const LABEL_MAP = {
  0: "Benign",
  1: "Malignant",

  B: "Benign",
  M: "Malignant",
};


// ============================================================
// HEART DISEASE LABEL MAP
// ============================================================

export const HEART_LABEL_MAP = {
  0: "No Disease",
  1: "Disease",
};


// ============================================================
// SAMPLE BREAST CANCER INPUT
//
// Based on the WDBC feature structure.
// Values are strings because they are used directly
// inside frontend input fields.
// ============================================================

export const SAMPLE_INPUT = {
  // Mean

  radius_mean: "17.99",
  texture_mean: "10.38",
  perimeter_mean: "122.8",
  area_mean: "1001",
  smoothness_mean: "0.1184",
  compactness_mean: "0.2776",
  concavity_mean: "0.3001",
  concave_points_mean: "0.1471",
  symmetry_mean: "0.2419",
  fractal_dimension_mean: "0.07871",

  // Standard Error

  radius_se: "1.095",
  texture_se: "0.9053",
  perimeter_se: "8.589",
  area_se: "153.4",
  smoothness_se: "0.006399",
  compactness_se: "0.04904",
  concavity_se: "0.05373",
  concave_points_se: "0.01587",
  symmetry_se: "0.03003",
  fractal_dimension_se: "0.006193",

  // Worst

  radius_worst: "25.38",
  texture_worst: "17.33",
  perimeter_worst: "184.6",
  area_worst: "2019",
  smoothness_worst: "0.1622",
  compactness_worst: "0.6656",
  concavity_worst: "0.7119",
  concave_points_worst: "0.2654",
  symmetry_worst: "0.4601",
  fractal_dimension_worst: "0.1189",
};


// ============================================================
// SAMPLE HEART DISEASE INPUT
// ============================================================

export const SAMPLE_HEART_INPUT = {
  age: "55",
  sex: "Male",
  cp: "typical angina",
  trestbps: "140",
  chol: "250",
  fbs: "0",
  restecg: "normal",
  thalch: "150",
  exang: "0",
  oldpeak: "1.0",
  slope: "upsloping",
  ca: "0",
  thal: "normal",
};


// ============================================================
// HEART DISEASE SELECT OPTIONS
//
// These values correspond to the categorical preprocessing
// performed by the backend.
// ============================================================

export const HEART_SELECT_OPTIONS = {
  sex: [
    {
      value: "Male",
      label: "Male",
    },
    {
      value: "Female",
      label: "Female",
    },
  ],

  cp: [
    {
      value: "typical angina",
      label: "Typical Angina",
    },
    {
      value: "atypical angina",
      label: "Atypical Angina",
    },
    {
      value: "non-anginal pain",
      label: "Non-anginal Pain",
    },
    {
      value: "asymptomatic",
      label: "Asymptomatic",
    },
  ],

  fbs: [
    {
      value: "0",
      label: "No",
    },
    {
      value: "1",
      label: "Yes",
    },
  ],

  restecg: [
    {
      value: "normal",
      label: "Normal",
    },
    {
      value: "st-t abnormality",
      label: "ST-T Wave Abnormality",
    },
    {
      value: "lv hypertrophy",
      label: "Left Ventricular Hypertrophy",
    },
  ],

  exang: [
    {
      value: "0",
      label: "No",
    },
    {
      value: "1",
      label: "Yes",
    },
  ],

  slope: [
    {
      value: "upsloping",
      label: "Upsloping",
    },
    {
      value: "flat",
      label: "Flat",
    },
    {
      value: "downsloping",
      label: "Downsloping",
    },
  ],

  thal: [
    {
      value: "normal",
      label: "Normal",
    },
    {
      value: "fixed defect",
      label: "Fixed Defect",
    },
    {
      value: "reversable defect",
      label: "Reversable Defect",
    },
  ],
};


// ============================================================
// MODEL METADATA
//
// Used by the frontend to display information about the
// selected model.
// ============================================================

export const MODEL_INFO = {
  hybrid_vqc: {
    name: "Hybrid Quantum-Classical VQC",
    shortName: "Hybrid VQC",
    description:
      "A hybrid quantum-classical variational quantum classifier.",
    type: "hybrid",
    qubits: 4,
  },

  svm: {
    name: "Support Vector Machine",
    shortName: "SVM",
    description:
      "A classical Support Vector Machine baseline.",
    type: "classical",
    qubits: null,
  },
};


// ============================================================
// DATASET INFORMATION
// ============================================================

export const DATASET_INFO = {
  breast_cancer: {
    id: "breast_cancer",
    name: "Wisconsin Breast Cancer",
    shortName: "Breast Cancer",
    featureCount: 30,
    classes: [
      "Benign",
      "Malignant",
    ],
  },

  heart_disease: {
    id: "heart_disease",
    name: "UCI Heart Disease",
    shortName: "Heart Disease",
    featureCount: 13,
    classes: [
      "No Disease",
      "Disease",
    ],
  },
};


// ============================================================
// DEFAULT VALUES
// ============================================================

export const DEFAULT_DISEASE = "breast_cancer";

export const DEFAULT_MODEL = "hybrid_vqc";