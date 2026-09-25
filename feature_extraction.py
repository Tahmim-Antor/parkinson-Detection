"""
feature_extraction.py

Reusable feature-extraction module for the Parkinson's Voice Detection
project.

Every function in this file is a direct, unmodified port of the logic in
the originally-provided dataset-generation script that produced
data1.csv. Nothing about the audio loading, framing, or the 22 feature
formulas has been changed. The only things adapted are:

  - packaging the code into importable functions instead of a
    top-of-file script that walks a folder of WAV files
  - `extract_features(audio_path)`, a thin wrapper around the original
    `extract_22_features` that validates the output against the
    authoritative FEATURE_ORDER defined in config.py

Use `extract_features(audio_path)` from app.py / train_model.py.
"""

import warnings

import numpy as np
from scipy import signal
from scipy.signal import find_peaks

from config import FEATURE_ORDER

warnings.filterwarnings("ignore")

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "librosa is required for feature extraction. "
        "Install it with: pip install librosa"
    ) from exc


# ============================================================
# LOAD AND PREPROCESS AUDIO
# ============================================================

def load_audio(filepath):
    """Load a WAV file, remove DC offset and linear trend, normalize peak."""

    # Load audio
    y, sr = librosa.load(filepath, sr=None, mono=True)

    y = np.asarray(y, dtype=np.float64)

    # Remove DC component
    y = y - np.mean(y)

    # Remove linear trend
    y = signal.detrend(y)

    # Normalize amplitude
    peak = np.max(np.abs(y))

    if peak > 0:
        y = y / peak

    return y, sr


# ============================================================
# EXTRACT FUNDAMENTAL FREQUENCY F0
# ============================================================

def get_f0(y, sr):

    frame_length = int(0.040 * sr)
    hop_length = int(0.010 * sr)

    # librosa.pyin frame length should be even
    if frame_length % 2 != 0:
        frame_length += 1

    # Too short recording
    if len(y) < frame_length:
        return np.array([])

    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=60,
            fmax=400,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        # Keep only valid voiced F0 values
        f0 = f0[np.isfinite(f0) & (f0 > 0)]

        return f0

    except Exception:
        return np.array([])


# ============================================================
# EXTRACT PERIODS AND AMPLITUDES
# ============================================================

def get_periods_amplitudes(y, sr, f0_mean):

    if not np.isfinite(f0_mean):
        return np.array([]), np.array([])

    if f0_mean <= 0:
        return np.array([]), np.array([])

    # Bandpass around F0
    low = max(40, 0.7 * f0_mean)
    high = min(1.4 * f0_mean, 0.45 * sr)

    if high <= low:
        return np.array([]), np.array([])

    try:
        b, a = signal.butter(
            4, [low / (sr / 2), high / (sr / 2)], btype="bandpass"
        )
        yf = signal.filtfilt(b, a, y)
    except Exception:
        yf = y

    # Minimum distance between peaks
    distance = max(2, int(0.7 * sr / f0_mean))

    peaks, _ = find_peaks(yf, distance=distance)

    if len(peaks) < 3:
        return np.array([]), np.array([])

    nominal_period = 1 / f0_mean

    periods = []
    amplitudes = []

    for i in range(len(peaks) - 1):

        period = (peaks[i + 1] - peaks[i]) / sr

        # Keep plausible cycles
        if 0.7 * nominal_period < period < 1.4 * nominal_period:

            segment = y[peaks[i]:peaks[i + 1]]

            if len(segment) > 1:
                periods.append(period)
                amplitude = np.max(segment) - np.min(segment)
                amplitudes.append(amplitude)

    return np.array(periods), np.array(amplitudes)


# ============================================================
# JITTER FEATURES
# ============================================================

def calculate_jitter(periods):

    if len(periods) < 6:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    mean_period = np.mean(periods)

    if mean_period <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Jitter absolute
    jitter_abs = np.mean(np.abs(np.diff(periods)))

    # Jitter percentage
    jitter_percent = (jitter_abs / mean_period) * 100

    # RAP
    rap_values = []
    for i in range(1, len(periods) - 1):
        local_mean = np.mean(periods[i - 1:i + 2])
        rap_values.append(abs(periods[i] - local_mean))

    rap = (np.mean(rap_values) / mean_period) if len(rap_values) > 0 else np.nan

    # PPQ5
    ppq_values = []
    for i in range(2, len(periods) - 2):
        local_mean = np.mean(periods[i - 2:i + 3])
        ppq_values.append(abs(periods[i] - local_mean))

    ppq = (np.mean(ppq_values) / mean_period) if len(ppq_values) > 0 else np.nan

    # DDP = 3 x RAP
    ddp = 3 * rap if np.isfinite(rap) else np.nan

    return jitter_percent, jitter_abs, rap, ppq, ddp


# ============================================================
# SHIMMER FEATURES
# ============================================================

def calculate_shimmer(amplitudes):

    if len(amplitudes) < 6:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    mean_amplitude = np.mean(amplitudes)

    if mean_amplitude <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # MDVP Shimmer
    shimmer = np.mean(np.abs(np.diff(amplitudes))) / mean_amplitude

    # Shimmer dB
    shimmer_db_values = []
    for i in range(len(amplitudes) - 1):
        a1 = max(amplitudes[i], 1e-12)
        a2 = max(amplitudes[i + 1], 1e-12)
        value = abs(20 * np.log10(a2 / a1))
        shimmer_db_values.append(value)

    shimmer_db = np.mean(shimmer_db_values)

    # APQ3
    apq3_values = []
    for i in range(1, len(amplitudes) - 1):
        local_mean = np.mean(amplitudes[i - 1:i + 2])
        apq3_values.append(abs(amplitudes[i] - local_mean))

    apq3 = (np.mean(apq3_values) / mean_amplitude) if len(apq3_values) > 0 else np.nan

    # APQ5
    apq5_values = []
    for i in range(2, len(amplitudes) - 2):
        local_mean = np.mean(amplitudes[i - 2:i + 3])
        apq5_values.append(abs(amplitudes[i] - local_mean))

    apq5 = (np.mean(apq5_values) / mean_amplitude) if len(apq5_values) > 0 else np.nan

    # APQ11
    apq_values = []
    if len(amplitudes) >= 11:
        for i in range(5, len(amplitudes) - 5):
            local_mean = np.mean(amplitudes[i - 5:i + 6])
            apq_values.append(abs(amplitudes[i] - local_mean))

    apq = (np.mean(apq_values) / mean_amplitude) if len(apq_values) > 0 else np.nan

    # DDA = 3 x APQ3
    dda = 3 * apq3 if np.isfinite(apq3) else np.nan

    return shimmer, shimmer_db, apq3, apq5, apq, dda


# ============================================================
# HNR AND NHR
# ============================================================

def calculate_hnr_nhr(y, sr, f0_mean):

    if not np.isfinite(f0_mean):
        return np.nan, np.nan

    if f0_mean <= 0:
        return np.nan, np.nan

    # Remove mean
    y = y - np.mean(y)

    # Autocorrelation
    autocorr = signal.correlate(y, y, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]

    if autocorr[0] <= 0:
        return np.nan, np.nan

    autocorr = autocorr / autocorr[0]

    # Pitch period in samples
    pitch_period = int(round(sr / f0_mean))

    low_lag = max(1, int(0.8 * pitch_period))
    high_lag = min(len(autocorr) - 1, int(1.2 * pitch_period))

    if high_lag <= low_lag:
        return np.nan, np.nan

    r = np.max(autocorr[low_lag:high_lag + 1])
    r = np.clip(r, 1e-6, 1 - 1e-6)

    # Harmonics to Noise Ratio
    hnr = 10 * np.log10(r / (1 - r))

    # Noise to Harmonics Ratio
    nhr = (1 - r) / r

    return nhr, hnr


# ============================================================
# RPDE APPROXIMATION
# ============================================================

def calculate_rpde(periods):

    if len(periods) < 20:
        return np.nan

    std_value = np.std(periods)

    if std_value == 0:
        return np.nan

    x = (periods - np.mean(periods)) / std_value

    hist, _ = np.histogram(x, bins=20)
    hist = hist[hist > 0]

    if len(hist) < 2:
        return np.nan

    probabilities = hist / np.sum(hist)

    entropy = -np.sum(probabilities * np.log(probabilities))

    rpde = entropy / np.log(len(probabilities))

    return rpde


# ============================================================
# DFA
# ============================================================

def calculate_dfa(x):

    x = np.asarray(x)
    x = x[np.isfinite(x)]

    if len(x) < 30:
        return np.nan

    # Integrated signal
    y = np.cumsum(x - np.mean(x))

    max_scale = len(y) // 4

    if max_scale < 8:
        return np.nan

    scales = np.unique(
        np.logspace(np.log10(4), np.log10(max_scale), 12).astype(int)
    )

    fluctuation = []
    valid_scales = []

    for scale_value in scales:

        number_segments = len(y) // scale_value

        if number_segments < 2:
            continue

        rms_values = []

        for j in range(number_segments):

            segment = y[j * scale_value:(j + 1) * scale_value]

            t = np.arange(scale_value)

            coefficients = np.polyfit(t, segment, 1)

            trend = np.polyval(coefficients, t)

            rms = np.sqrt(np.mean((segment - trend) ** 2))

            rms_values.append(rms)

        F = np.sqrt(np.mean(np.array(rms_values) ** 2))

        if F > 0:
            fluctuation.append(F)
            valid_scales.append(scale_value)

    if len(valid_scales) < 3:
        return np.nan

    alpha = np.polyfit(np.log(valid_scales), np.log(fluctuation), 1)[0]

    return alpha


# ============================================================
# SPREAD1 AND SPREAD2 APPROXIMATION
# ============================================================

def calculate_spread(f0):

    if len(f0) < 10:
        return np.nan, np.nan

    x = np.log(f0)

    spread1 = np.std(x)
    spread2 = np.std(np.diff(x))

    return spread1, spread2


# ============================================================
# D2 APPROXIMATION
# ============================================================

def calculate_d2(x):

    if len(x) < 50:
        return np.nan

    std_value = np.std(x)

    if std_value == 0:
        return np.nan

    x = (x - np.mean(x)) / std_value

    # Limit computation
    if len(x) > 500:
        x = x[:500]

    points = np.column_stack((x[:-1], x[1:]))

    distances = []

    for i in range(len(points)):
        d = np.linalg.norm(points[i + 1:] - points[i], axis=1)
        d = d[d > 0]
        distances.extend(d)

    distances = np.array(distances)

    if len(distances) < 20:
        return np.nan

    radii = np.percentile(distances, [10, 20, 30, 40])

    correlation = []
    for radius in radii:
        value = np.mean(distances < radius)
        correlation.append(value)

    correlation = np.array(correlation)

    valid = (correlation > 0) & (correlation < 1)

    if np.sum(valid) < 2:
        return np.nan

    d2 = np.polyfit(np.log(radii[valid]), np.log(correlation[valid]), 1)[0]

    return d2


# ============================================================
# PPE APPROXIMATION
# ============================================================

def calculate_ppe(f0):

    if len(f0) < 20:
        return np.nan

    x = np.log(f0)
    x = x - np.median(x)

    hist, _ = np.histogram(x, bins=20)
    hist = hist[hist > 0]

    if len(hist) == 0:
        return np.nan

    probabilities = hist / np.sum(hist)

    ppe = -np.sum(probabilities * np.log(probabilities))

    return ppe


# ============================================================
# MAIN 22-FEATURE EXTRACTION FUNCTION (unmodified logic)
# ============================================================

def extract_22_features(filepath):
    """
    Extract the same 22 UCI-style acoustic features as the original
    dataset-generation script. Returns a dict keyed by the original
    feature names. Values are np.nan for any feature that cannot be
    computed on this particular recording (e.g. too short / unvoiced),
    exactly like the original script.
    """

    y, sr = load_audio(filepath)

    f0 = get_f0(y, sr)

    # If F0 extraction fails
    if len(f0) < 5:
        return {feature: np.nan for feature in FEATURE_ORDER}

    fo = np.mean(f0)
    fhi = np.max(f0)
    flo = np.min(f0)

    periods, amplitudes = get_periods_amplitudes(y, sr, fo)

    jitter_percent, jitter_abs, rap, ppq, ddp = calculate_jitter(periods)

    shimmer, shimmer_db, apq3, apq5, apq, dda = calculate_shimmer(amplitudes)

    nhr, hnr = calculate_hnr_nhr(y, sr, fo)

    rpde = calculate_rpde(periods)
    dfa = calculate_dfa(f0)
    spread1, spread2 = calculate_spread(f0)
    d2 = calculate_d2(f0)
    ppe = calculate_ppe(f0)

    features = {
        "MDVP:Fo(Hz)": fo,
        "MDVP:Fhi(Hz)": fhi,
        "MDVP:Flo(Hz)": flo,
        "MDVP:Jitter(%)": jitter_percent,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp,
        "MDVP:Shimmer": shimmer,
        "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5,
        "MDVP:APQ": apq,
        "Shimmer:DDA": dda,
        "NHR": nhr,
        "HNR": hnr,
        "RPDE": rpde,
        "DFA": dfa,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe,
    }

    return features


# ============================================================
# PUBLIC API used by app.py / train_model.py
# ============================================================

def extract_features(audio_path):
    """
    Extract the 22 acoustic features from a WAV file and return them as
    an ordered list matching config.FEATURE_ORDER exactly.

    Returns
    -------
    dict
        {"features_dict": {name: value, ...},   # for display
         "ordered_values": [v1, v2, ..., v22]}   # for the model, in
                                                   # FEATURE_ORDER

    Raises
    ------
    ValueError
        If the number of extracted features does not equal 22, or the
        keys do not match FEATURE_ORDER exactly.
    """

    features_dict = extract_22_features(audio_path)

    if len(features_dict) != 22:
        raise ValueError(
            f"Feature extraction returned {len(features_dict)} features, "
            f"expected exactly 22."
        )

    if set(features_dict.keys()) != set(FEATURE_ORDER):
        raise ValueError(
            "Extracted feature names do not match the authoritative "
            "FEATURE_ORDER defined in config.py."
        )

    ordered_values = [features_dict[name] for name in FEATURE_ORDER]

    return {
        "features_dict": features_dict,
        "ordered_values": ordered_values,
    }
