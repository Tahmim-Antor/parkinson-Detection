"""
config.py

Central configuration for the Parkinson's Voice Detection project.

This is the SINGLE authoritative source for:
- the 22-feature order (must exactly match the order used to build data1.csv)
- dataset / model file paths
- class-label mapping
- SVM / cross-validation settings (mirrored from the original training
  script -- NOT re-invented here)

Do not duplicate the feature list anywhere else in the project. Every
other module imports FEATURE_ORDER from here.
"""

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "data1.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "parkinson_svm_pipeline.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# ----------------------------------------------------------------------
# Non-feature identifier columns present in data1.csv
# ----------------------------------------------------------------------
# NOTE ON A CORRECTED BUG:
# The originally-provided training script only dropped "subject" and
# "status" before building X:
#
#     X = df.drop(columns=["subject", "status"])
#
# Because data1.csv's columns are ["name", "subject"] + 22 features +
# ["status"], this left the "name" column (a recording filename string)
# inside the feature matrix -- meaning the original script was actually
# training on 23 columns, one of which is a non-numeric identifier, not
# an acoustic measurement. That is a bug, not a deliberate feature
# choice, so it has been corrected here by excluding BOTH identifier
# columns. This is the only behavioral deviation this project makes
# from the code you provided. See README.md for details.
ID_COLUMNS = ["name", "subject"]
TARGET_COLUMN = "status"

# ----------------------------------------------------------------------
# The 22 acoustic features, in the EXACT order produced by the provided
# dataset-generation script (see `feature_names` / `extract_22_features`
# in the original script).
# ----------------------------------------------------------------------
FEATURE_ORDER = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

assert len(FEATURE_ORDER) == 22, "FEATURE_ORDER must contain exactly 22 features."

# ----------------------------------------------------------------------
# Class labels (dataset generator: Parkinson's = 1, Healthy = 0)
# ----------------------------------------------------------------------
CLASS_LABELS = {
    0: "Healthy",
    1: "Parkinson's",
}
# index 0 -> Healthy, index 1 -> Parkinson's (matches status encoding)
TARGET_NAMES_ORDERED = [CLASS_LABELS[0], CLASS_LABELS[1]]

# ----------------------------------------------------------------------
# Known data-quality fix already implicit in the training script: a
# sentinel value of 999999 appears in NHR and must be treated as missing
# (NaN) so the imputer -- not the raw sentinel -- handles it.
# ----------------------------------------------------------------------
NHR_SENTINEL_VALUE = 999999

# ----------------------------------------------------------------------
# SVM / cross-validation configuration.
# Mirrors the provided training script exactly. Do not change these
# values without first changing the source training methodology.
# ----------------------------------------------------------------------
SVM_KERNEL = "rbf"
SVM_C = 10
SVM_GAMMA = "scale"
SVM_CLASS_WEIGHT = "balanced"

# The original SVC(...) does NOT set probability=True, so predict_proba()
# is not available on the trained model as provided. Flip this to True
# only if you deliberately retrain with probability=True in
# train_model.py -- doing so changes the fitted model (adds internal
# Platt-scaling calibration via extra internal CV), so it is left False
# by default to preserve your original model exactly as given.
PROBABILITY_ENABLED = False

CV_N_SPLITS = 5
CV_SHUFFLE = True
RANDOM_STATE = 42

IMPUTER_STRATEGY = "median"
