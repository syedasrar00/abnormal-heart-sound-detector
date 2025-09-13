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
# Audio Recording Function
# -----------------------
def record_audio(duration=5, fs=22050):
    """Record audio from microphone"""
    st.info("Recording... Speak or place stethoscope near mic")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    st.success("Recording complete!")
    return recording.flatten(), fs

# -----------------------
# Feature Extraction
# -----------------------
def extract_features(y, sr):
    """Extract MFCCs + other statistics"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    features = np.concatenate([mfccs_mean, mfccs_std])
    return features

# -----------------------
# Streamlit UI
# -----------------------
st.title("🩺 Heart Sound Classification")
st.write("Record your heart sound and classify it as **Normal** or **Abnormal**.")

duration = st.slider("Select Recording Duration (seconds)", 3, 10, 5)

if st.button("Record Heart Sound"):
    y, sr = record_audio(duration=duration)
    st.audio(y, format="audio/wav", sample_rate=sr)

    # Extract features
    features = extract_features(y, sr)
    features_scaled = scaler.transform([features])

    # Predict
    pred = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    # Output
    st.subheader("🔍 Prediction Result")
    if pred == 0:
        st.success(f"✅ Normal Heart Sound (Confidence: {proba.max()*100:.2f}%)")
    else:
        st.error(f"⚠️ Abnormal Heart Sound (Confidence: {proba.max()*100:.2f}%)")

