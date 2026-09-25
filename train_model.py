"""
train_model.py

Trains the SVM classifier using the SAME methodology as the originally
provided training script:

  - StandardScaler + SimpleImputer(median) + SVC(rbf, C=10, gamma="scale",
    class_weight="balanced") in a single sklearn Pipeline
  - StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
    grouped by subject, so no subject appears in both train and test
    within a fold
  - NHR sentinel value (999999) treated as missing before imputation
  - Final model refit on the FULL dataset for deployment, saved as one
    joblib pipeline

The only behavioral correction versus the originally-provided script is
dropping the "name" identifier column from the feature matrix in
addition to "subject" and "status" (see config.py for why -- the
original script left "name" inside X, which is a bug).

Run:
    python train_model.py
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config


def load_dataset():
    """Load data1.csv and validate it has every expected column."""

    if not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at: {config.DATA_PATH}\n"
            f"Place your generated dataset there (see README.md)."
        )

    df = pd.read_csv(config.DATA_PATH)

    required_columns = set(config.ID_COLUMNS + config.FEATURE_ORDER + [config.TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"data1.csv is missing expected columns: {sorted(missing_columns)}\n"
            f"Expected columns: {sorted(required_columns)}"
        )

    return df


def main():

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    df = load_dataset()

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    # --------------------------------------------------------------
    # BASIC DATA ANALYSIS (same checks as the original training script)
    # --------------------------------------------------------------

    print("\n==============================")
    print("DATASET ANALYSIS")
    print("==============================")

    print("Number of recordings:", len(df))
    print("Number of subjects:", df["subject"].nunique())

    print("\nClass distribution:")
    print(df[config.TARGET_COLUMN].value_counts())

    print("\nMissing values (before NHR sentinel fix):")
    print(df.isnull().sum())

    print("\nRecordings per subject:")
    print(df.groupby("subject").size().describe())

    subject_status = df.groupby("subject")[config.TARGET_COLUMN].nunique()
    subjects_with_mixed_status = int((subject_status > 1).sum())

    print("\nSubjects having more than one status (should be 0):")
    print(subjects_with_mixed_status)

    if subjects_with_mixed_status > 0:
        print(
            "WARNING: some subjects have recordings under more than one "
            "class label. This can leak information across folds."
        )

    # --------------------------------------------------------------
    # FIX KNOWN SENTINEL VALUE (NHR == 999999 -> NaN)
    # --------------------------------------------------------------

    df["NHR"] = df["NHR"].replace(config.NHR_SENTINEL_VALUE, np.nan)


    # --------------------------------------------------------------
    # SEPARATE FEATURES, TARGET, GROUPS
    #
    # CORRECTED vs. the original script: both "name" and "subject" are
    # dropped here (the original script only dropped "subject", which
    # silently left "name" inside the feature matrix).
    # --------------------------------------------------------------

    X = df[config.FEATURE_ORDER].copy()  # guarantees FEATURE_ORDER column order
    y = df[config.TARGET_COLUMN]
    groups = df["subject"]

    if list(X.columns) != config.FEATURE_ORDER:
        raise ValueError("Feature order mismatch when building X from data1.csv.")

    print("\n==============================")
    print("FEATURE INFORMATION")
    print("==============================")
    print("Number of features:", X.shape[1])
    print("\nFeatures:")
    print(list(X.columns))

    if X.shape[1] != 22:
        raise ValueError(f"Expected exactly 22 features, got {X.shape[1]}.")

    # --------------------------------------------------------------
    # BUILD PIPELINE (identical to the original training script)
    # --------------------------------------------------------------

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=config.IMPUTER_STRATEGY)),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=config.SVM_KERNEL,
            C=config.SVM_C,
            gamma=config.SVM_GAMMA,
            class_weight=config.SVM_CLASS_WEIGHT,
            probability=config.PROBABILITY_ENABLED,
        )),
    ])

    # --------------------------------------------------------------
    # SUBJECT-WISE STRATIFIED K-FOLD CROSS-VALIDATION
    # --------------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=config.CV_N_SPLITS,
        shuffle=config.CV_SHUFFLE,
        random_state=config.RANDOM_STATE,
    )

    fold_accuracies = []
    fold_leakage_checks = []
    all_y_true = []
    all_y_pred = []

    print("\n\n==============================")
    print(f"{config.CV_N_SPLITS}-FOLD SUBJECT-WISE SVM CROSS-VALIDATION")
    print("==============================")

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        train_subjects = set(groups.iloc[train_idx])
        test_subjects = set(groups.iloc[test_idx])
        common_subjects = train_subjects.intersection(test_subjects)

        print(f"\n------------------------------")
        print(f"Fold {fold}")
        print(f"------------------------------")
        print("Training subjects:", len(train_subjects))
        print("Testing subjects :", len(test_subjects))
        print("Subjects appearing in both train and test:", common_subjects)

        if common_subjects:
            raise RuntimeError(
                f"Subject leakage detected in fold {fold}: {common_subjects}"
            )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        fold_accuracies.append(accuracy)
        fold_leakage_checks.append(len(common_subjects) == 0)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print("Fold accuracy:", round(accuracy, 4))

    # --------------------------------------------------------------
    # FINAL CROSS-VALIDATION RESULTS
    # --------------------------------------------------------------

    print("\n\n==============================")
    print("FINAL CROSS-VALIDATION RESULT")
    print("==============================")

    for i, accuracy in enumerate(fold_accuracies, start=1):
        print(f"Fold {i} accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    mean_accuracy = float(np.mean(fold_accuracies))
    std_accuracy = float(np.std(fold_accuracies))

    print("\nMean accuracy:", round(mean_accuracy, 4))
    print("Mean accuracy (%):", round(mean_accuracy * 100, 2), "%")
    print("Standard deviation:", round(std_accuracy, 4))
    print("Standard deviation (%):", round(std_accuracy * 100, 2), "%")

    print("\n\n==============================")
    print("CLASSIFICATION REPORT (pooled out-of-fold predictions)")
    print("==============================")

    report_text = classification_report(
        all_y_true, all_y_pred,
        target_names=config.TARGET_NAMES_ORDERED,
        digits=4,
    )
    print(report_text)

    report_dict = classification_report(
        all_y_true, all_y_pred,
        target_names=config.TARGET_NAMES_ORDERED,
        digits=4,
        output_dict=True,
    )

    cm = confusion_matrix(all_y_true, all_y_pred)

    print("\n==============================")
    print("CONFUSION MATRIX (pooled out-of-fold predictions)")
    print("==============================")
    print(cm)
    print("\nRows = Actual, Columns = Predicted")
    print("\n              Predicted")
    print("              Healthy   Parkinson's")
    print(f"Actual Healthy    {cm[0, 0]:5d}      {cm[0, 1]:5d}")
    print(f"Actual PD         {cm[1, 0]:5d}      {cm[1, 1]:5d}")

    # --------------------------------------------------------------
    # TRAIN FINAL MODEL ON THE FULL DATASET (for deployment)
    #
    # This is a SEPARATE fit from the CV loop above. The numbers
    # reported to the user (accuracy, fold scores, confusion matrix)
    # come ONLY from the cross-validation loop, never from this final
    # fit -- that would be training accuracy, not CV accuracy.
    # --------------------------------------------------------------

    print("\n\n==============================")
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("==============================")

    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=config.IMPUTER_STRATEGY)),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=config.SVM_KERNEL,
            C=config.SVM_C,
            gamma=config.SVM_GAMMA,
            class_weight=config.SVM_CLASS_WEIGHT,
            probability=config.PROBABILITY_ENABLED,
        )),
    ])

    final_pipeline.fit(X, y)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(final_pipeline, config.MODEL_PATH)
    print(f"Saved final trained pipeline to: {config.MODEL_PATH}")

    # --------------------------------------------------------------
    # SAVE METADATA FOR THE STREAMLIT APP
    # --------------------------------------------------------------

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_recordings": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_features": int(X.shape[1]),
        "feature_order": config.FEATURE_ORDER,
        "class_labels": config.CLASS_LABELS,
        "model": {
            "algorithm": "Support Vector Machine (SVC)",
            "kernel": config.SVM_KERNEL,
            "C": config.SVM_C,
            "gamma": config.SVM_GAMMA,
            "class_weight": config.SVM_CLASS_WEIGHT,
            "probability_enabled": config.PROBABILITY_ENABLED,
            "imputer_strategy": config.IMPUTER_STRATEGY,
            "scaler": "StandardScaler",
        },
        "cross_validation": {
            "strategy": "StratifiedGroupKFold (subject-wise)",
            "n_splits": config.CV_N_SPLITS,
            "shuffle": config.CV_SHUFFLE,
            "random_state": config.RANDOM_STATE,
            "fold_accuracies": [float(a) for a in fold_accuracies],
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "no_subject_leakage_in_any_fold": bool(all(fold_leakage_checks)),
        },
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }

    with open(config.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model metadata to: {config.METADATA_PATH}")

    print("\n\n==============================")
    print("SUMMARY")
    print("==============================")
    print("Dataset recordings :", len(df))
    print("Subjects            :", df["subject"].nunique())
    print("Features            :", X.shape[1])
    print(f"CV folds            : {config.CV_N_SPLITS}")
    print("SVM kernel          :", config.SVM_KERNEL)
    print("C                   :", config.SVM_C)
    print("Gamma               :", config.SVM_GAMMA)
    print("Scaler              : StandardScaler")
    print("Missing value method: Median imputation")
    print("Data splitting      : Subject-wise")
    print("Random state        :", config.RANDOM_STATE)
    print(f"\nFinal CV accuracy: {mean_accuracy * 100:.2f}% +/- {std_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
