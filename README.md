# Parkinson’s Disease Detection from Voice

This project uses voice recordings and machine learning to detect Parkinson’s disease based on acoustic features.

## Main Workflow

- Audio preprocessing and denoising
- Voice feature extraction
- Feature analysis and visualization
- Subject-wise data handling
- SVM classification
- 5-fold subject-wise cross-validation

## Features Used

The project analyzes features such as:

- Jitter
- Shimmer
- RPDE
- D2
- PPE
- Fundamental frequency-related features

## Model

A Support Vector Machine (SVM) classifier is used with:

- Missing-value handling
- Feature standardization
- `StratifiedGroupKFold`

Subject-wise grouping is used so that recordings from the same person do not appear in both training and testing sets.

## Technologies

- Python
- NumPy
- Pandas
- SciPy
- Librosa
- PyWavelets
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

The main dataset is stored in:

```text
data1.csv
```

It contains multiple voice recordings for different subjects along with their Parkinson’s disease status.

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Disclaimer

This project is developed for academic and research purposes only and is not intended for clinical diagnosis.
