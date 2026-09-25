"""
app.py

Streamlit application for Parkinson's Voice Detection.

Uses ONLY:
- the feature extraction logic in feature_extraction.py (identical math
  to the originally-provided dataset-generation script)
- the trained pipeline saved by train_model.py (imputer + scaler + SVM,
  fitted entirely on the training dataset -- never refit on an
  uploaded recording)
- the real cross-validation results computed by train_model.py

No numbers on this page are invented. If train_model.py has not been
run yet, the app tells the user to run it instead of guessing at
results.
"""

import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import config
from feature_extraction import extract_features


st.set_page_config(
    page_title="Parkinson's Voice Detection",
    page_icon="🎙️",
    layout="centered",
)


# ----------------------------------------------------------------------
# CACHED LOADERS
# ----------------------------------------------------------------------

@st.cache_resource
def load_pipeline():
    if not os.path.exists(config.MODEL_PATH):
        return None
    return joblib.load(config.MODEL_PATH)


@st.cache_data
def load_metadata():
    if not os.path.exists(config.METADATA_PATH):
        return None
    with open(config.METADATA_PATH, "r") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------

st.title("Parkinson's Voice Detection")
st.caption("Voice-based Parkinson's classification using acoustic features and SVM")



pipeline = load_pipeline()
metadata = load_metadata()

if pipeline is None or metadata is None:
    st.error(
        "No trained model was found.\n\n"
        f"Expected files:\n"
        f"- `{config.MODEL_PATH}`\n"
        f"- `{config.METADATA_PATH}`\n\n"
        "Please train the model first by running:\n\n"
        "```bash\npython train_model.py\n```"
    )
    st.stop()


# ----------------------------------------------------------------------
# SECTION 1 -- UPLOAD VOICE
# ----------------------------------------------------------------------

st.header("1. Upload Voice Recording")

uploaded_file = st.file_uploader(
    "Upload a WAV recording",
    type=["wav"],
)

if uploaded_file is not None:
    st.write(f"**File name:** {uploaded_file.name}")
    st.audio(uploaded_file, format="audio/wav")

analyze_clicked = st.button(
    "Analyze Voice",
    type="primary",
    disabled=uploaded_file is None,
)

if uploaded_file is None:
    st.info("Upload a .wav file to enable analysis.")

# ----------------------------------------------------------------------
# ANALYSIS PIPELINE (only runs after the button is pressed)
# ----------------------------------------------------------------------

if analyze_clicked and uploaded_file is not None:

    # --- Basic file validation -----------------------------------
    if not uploaded_file.name.lower().endswith(".wav"):
        st.error("Unable to process this audio file.\nPlease upload a valid WAV recording.")
        st.stop()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
    except Exception:
        st.error("Unable to process this audio file.\nPlease upload a valid WAV recording.")
        st.stop()

    # --- Feature extraction ---------------------------------------
    st.header("2. Extracted Features")

    try:
        result = extract_features(tmp_path)
    except ValueError as e:
        st.error(f"Feature extraction failed.\nPlease check that the recording contains valid voice data.\n\nDetails: {e}")
        st.stop()
    except Exception:
        st.error("Feature extraction failed.\nPlease check that the recording contains valid voice data.")
        st.stop()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    features_dict = result["features_dict"]
    ordered_values = result["ordered_values"]

    # Explicit safety checks (per project requirements)
    if len(config.FEATURE_ORDER) != 22:
        st.error("Internal configuration error: FEATURE_ORDER does not contain 22 features.")
        st.stop()

    if len(ordered_values) != 22:
        st.error(
            f"Feature extraction returned {len(ordered_values)} features instead of 22. "
            "Prediction cannot proceed."
        )
        st.stop()

    feature_table = pd.DataFrame({
        "Feature": config.FEATURE_ORDER,
        "Value": ordered_values,
    })
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    # --- Missing value check ----------------------------------------
    st.header("3. Missing Values")

    missing_mask = [v is None or (isinstance(v, float) and np.isnan(v)) for v in ordered_values]
    has_missing = any(missing_mask)

    st.write(f"**Missing values detected:** {'Yes' if has_missing else 'No'}")

    if has_missing:
        missing_features = [name for name, is_missing in zip(config.FEATURE_ORDER, missing_mask) if is_missing]
        st.info(
            "The following features could not be computed from this recording: "
            f"{', '.join(missing_features)}.\n\n"
            "They will be filled in using the median value learned from the "
            "**training dataset** (the same imputer fitted during training). "
            "No new statistics are calculated from this single recording."
        )

    # --- Prediction ---------------------------------------------------
    st.header("4. Prediction")

    X_input = pd.DataFrame([ordered_values], columns=config.FEATURE_ORDER)

    if list(X_input.columns) != config.FEATURE_ORDER:
        st.error("Feature order mismatch. Prediction aborted.")
        st.stop()

    try:
        prediction = pipeline.predict(X_input)[0]
    except Exception as e:
        st.error(f"Prediction failed. Details: {e}")
        st.stop()

    predicted_label = config.CLASS_LABELS.get(int(prediction), str(prediction))

    if predicted_label == "Parkinson's":
        st.error(f"### Prediction: PARKINSON'S")
    else:
        st.success(f"### Prediction: HEALTHY")

    # --- Confidence / score -------------------------------------------
    st.header("5. Model Confidence")

    svm_step = pipeline.named_steps.get("svm")
    probability_available = bool(getattr(svm_step, "probability", False))

    if probability_available and hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_input)[0]
        healthy_p = proba[0] * 100
        pd_p = proba[1] * 100
        st.write(f"**Healthy probability:** {healthy_p:.1f}%")
        st.write(f"**Parkinson's probability:** {pd_p:.1f}%")
        st.caption(
            "These are model-estimated probabilities (Platt scaling), not a "
            "medically calibrated probability of disease."
        )
    else:
        decision_score = pipeline.decision_function(X_input)[0]
        st.write(f"**SVM decision function score:** {decision_score:.4f}")
        st.caption(
            "The underlying SVM was trained without `probability=True`, so no "
            "class probability is available. This number is the signed distance "
            "from the SVM decision boundary: positive values favor the "
            "'Parkinson's' class, negative values favor 'Healthy', and larger "
            "magnitude means the sample is farther from the boundary. "
            "It is a model confidence score, not a probability, and certainly "
            "not a medically calibrated one."
        )

    # --- Optional visualization -----------------------------------
    st.header("6. Feature Overview (optional)")
    st.caption(
        "Raw extracted feature values for this recording. This chart does not, "
        "by itself, indicate a diagnosis."
    )
    chart_df = feature_table.set_index("Feature")
    st.bar_chart(chart_df)


# ----------------------------------------------------------------------
# SECTION: MODEL INFORMATION (always visible)
# ----------------------------------------------------------------------

st.header("Model Information")

model_info = metadata["model"]
cv_info = metadata["cross_validation"]

col1, col2 = st.columns(2)

with col1:
    st.write(f"**Model:** {model_info['algorithm']}")
    st.write(f"**Kernel:** {model_info['kernel']}")
    st.write(f"**C:** {model_info['C']}")
    st.write(f"**Gamma:** {model_info['gamma']}")
    st.write(f"**Class weight:** {model_info['class_weight']}")

with col2:
    st.write(f"**Features used:** {metadata['n_features']}")
    st.write(f"**Cross-validation:** {cv_info['strategy']}")
    st.write(f"**CV folds:** {cv_info['n_splits']}")
    st.write(f"**CV accuracy:** {cv_info['mean_accuracy'] * 100:.2f}%")
    st.write(f"**Probability estimation enabled:** {model_info['probability_enabled']}")

st.caption(
    f"Trained on {metadata['n_recordings']} recordings from "
    f"{metadata['n_subjects']} subjects. Model trained at: {metadata['trained_at']}."
)


# ----------------------------------------------------------------------
# SECTION: CROSS-VALIDATION PERFORMANCE (always visible, real numbers)
# ----------------------------------------------------------------------

st.header("Cross-Validation Performance")

st.write(f"**Mean CV accuracy:** {cv_info['mean_accuracy'] * 100:.2f}%")
st.write(f"**Standard deviation:** {cv_info['std_accuracy'] * 100:.2f}%")

fold_df = pd.DataFrame({
    "Fold": [f"Fold {i + 1}" for i in range(len(cv_info["fold_accuracies"]))],
    "Accuracy (%)": [round(a * 100, 2) for a in cv_info["fold_accuracies"]],
})
st.dataframe(fold_df, use_container_width=True, hide_index=True)

if not cv_info.get("no_subject_leakage_in_any_fold", False):
    st.warning(
        "Warning: subject leakage was detected during training in at least one "
        "fold. Reported CV accuracy may be optimistic."
    )

with st.expander("Full classification report (pooled out-of-fold predictions)"):
    report = metadata["classification_report"]
    report_rows = []
    for label, stats in report.items():
        if isinstance(stats, dict):
            report_rows.append({
                "Class": label,
                "Precision": round(stats.get("precision", float("nan")), 4),
                "Recall": round(stats.get("recall", float("nan")), 4),
                "F1-score": round(stats.get("f1-score", float("nan")), 4),
                "Support": stats.get("support", ""),
            })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

with st.expander("Confusion matrix (pooled out-of-fold predictions)"):
    cm = metadata["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {name}" for name in config.TARGET_NAMES_ORDERED],
        columns=[f"Predicted: {name}" for name in config.TARGET_NAMES_ORDERED],
    )
    st.dataframe(cm_df, use_container_width=True)

st.caption(
    "All figures above come directly from train_model.py's cross-validation "
    "run and are not recalculated or estimated by this app."
)
