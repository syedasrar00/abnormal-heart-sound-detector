import streamlit as st
import sounddevice as sd
import numpy as np
import pickle
import librosa

# -----------------------
# Load Model and Scaler
# -----------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------
# Feature Functions
# -----------------------
def peak_amplitude(y):
    return np.max(np.abs(y))

def total_power_time(y):
    return np.sum(y**2)

def zero_crossing_rate(y):
    return np.mean(librosa.feature.zero_crossing_rate(y))

def peak_frequency(y, sr):
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    return freqs[np.argmax(spectrum)]

def peak_amplitude_freq(y, sr):
    spectrum = np.abs(np.fft.rfft(y))
    return np.max(spectrum)

def total_power_freq(y, sr):
    spectrum = np.abs(np.fft.rfft(y))
    return np.sum(spectrum**2)

def bandwidth(y, sr):
    return librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

def q_factor(y, sr):
    bw = bandwidth(y, sr)
    pf = peak_frequency(y, sr)
    return pf / bw if bw > 0 else 0

def cepstrum_peak_amplitude(y):
    spectrum = np.abs(np.fft.rfft(y))
    log_spectrum = np.log(spectrum + 1e-10)
    cepstrum = np.fft.irfft(log_spectrum)
    return np.max(np.abs(cepstrum))

def mean_stat(y):
    return np.mean(y)

def std_stat(y):
    return np.std(y)

def min_max_stat(y):
    return np.min(y), np.max(y)

def mfcc_features(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

# Feature extractor
def extract_all_features(y, sr):
    features = {}

    # Time domain
    features["peak_amplitude"] = peak_amplitude(y)
    features["total_power_time"] = total_power_time(y)
    features["zcr"] = zero_crossing_rate(y)

    # Frequency domain
    features["peak_frequency"] = peak_frequency(y, sr)
    features["peak_amplitude_freq"] = peak_amplitude_freq(y, sr)
    features["total_power_freq"] = total_power_freq(y, sr)
    features["bandwidth"] = bandwidth(y, sr)
    features["q_factor"] = q_factor(y, sr)

    # Cepstrum
    features["cepstrum_peak_amplitude"] = cepstrum_peak_amplitude(y)

    # Statistical
    features["mean"] = mean_stat(y)
    features["std"] = std_stat(y)
    features["min"], features["max"] = min_max_stat(y)

    # MFCCs (mfcc1 … mfcc13)
    mfccs = mfcc_features(y, sr, n_mfcc=13)
    for i, val in enumerate(mfccs, start=1):
        features[f"mfcc{i}"] = val

    return features

# -----------------------
# Streamlit UI
# -----------------------
st.title("🩺 Heart Sound Classification")
st.write("Record your heart sound and classify it as **Normal** or **Abnormal**.")

duration = st.slider("Select Recording Duration (seconds)", 3, 10, 5)

if st.button("Record Heart Sound"):
    st.info("Recording... Place stethoscope near microphone.")
    recording = sd.rec(int(duration * 22050), samplerate=22050, channels=1, dtype='float32')
    sd.wait()
    y = recording.flatten()
    sr = 22050

    st.audio(y, format="audio/wav", sample_rate=sr)

    # Extract features
    features = extract_all_features(y, sr)
    feature_vector = np.array(list(features.values())).reshape(1, -1)

    # Scale
    feature_vector_scaled = scaler.transform(feature_vector)

    # Predict
    pred = model.predict(feature_vector_scaled)[0]
    proba = model.predict_proba(feature_vector_scaled)[0]

    # Output
    st.subheader("🔍 Prediction Result")
    if pred == 0:
        st.success(f"✅ Normal Heart Sound (Confidence: {proba.max()*100:.2f}%)")
    else:
        st.error(f"⚠️ Abnormal Heart Sound (Confidence: {proba.max()*100:.2f}%)")
