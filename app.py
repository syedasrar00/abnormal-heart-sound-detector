import streamlit as st
import sounddevice as sd
import numpy as np
import pickle
import librosa
from io import BytesIO
import time
import noisereduce as nr

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Load Models and Scaler
# -----------------------
with open("model_base.pkl", "rb") as f:
    first_model = pickle.load(f)

with open("scaler_base.pkl", "rb") as f:
    first_scaler = pickle.load(f)
    
with open("model_main.pkl", "rb") as f:
    second_model = pickle.load(f)

with open("scaler_main.pkl", "rb") as f:
    second_scaler = pickle.load(f)

# -----------------------
# Audio Processing Functions
# -----------------------
def clip_audio(y, sr, max_duration=8):
    """
    Clip audio to specified duration from the middle if it's longer.
    
    Args:
        y (np.array): Input audio signal
        sr (int): Sampling rate
        max_duration (int): Maximum duration in seconds
        
    Returns:
        np.array: Clipped audio signal
    """
    # Calculate total duration
    total_duration = len(y) / sr
    
    if total_duration > max_duration:
        # Calculate middle point and samples needed
        middle_point = len(y) // 2
        samples_needed = int(max_duration * sr)
        half_samples = samples_needed // 2
        
        # Extract middle segment
        start_idx = middle_point - half_samples
        end_idx = middle_point + half_samples
        return y[start_idx:end_idx]
    return y

def reduce_noise(y, sr):
    """
    Reduce noise in the audio signal using the noisereduce library.
    
    Args:
        y (np.array): Input audio signal
        sr (int): Sampling rate
        
    Returns:
        np.array: Noise-reduced audio signal
    """
    # Reduce noise using the noise profile
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    return reduced_noise

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
# Modern Streamlit UI
# -----------------------

st.set_page_config(page_title="Heart Sound Classifier", page_icon="🩺", layout="centered")

# Initialize session state
if "ui_state" not in st.session_state:
    st.session_state.ui_state = "idle"  # idle, recording, uploaded, analyzing, result
    st.session_state.audio = None
    st.session_state.sr = 22050
    st.session_state.result = None
    st.session_state.duration = 5

# --- Helper Functions ---
def analyze_audio(audio_data, sr=22050):
    """Analyze audio data and return classification results"""
    # Clip audio to 8 seconds from middle if needed
    clipped_audio = clip_audio(audio_data, sr, max_duration=8)
    # Apply noise reduction
    clean_audio = reduce_noise(audio_data, sr)
    features = extract_all_features(clean_audio, sr)
    feature_vector = np.array(list(features.values())).reshape(1, -1)
    
    # First model - Heart sound detection
    feature_vector_scaled_one = first_scaler.transform(feature_vector)
    pred_one = first_model.predict(feature_vector_scaled_one)[0]
    proba_one = first_model.predict_proba(feature_vector_scaled_one)[0]
    if pred_one == 0:
        # Second model - Normal/Abnormal classification
        feature_vector_scaled_two = second_scaler.transform(feature_vector)
        pred_two = second_model.predict(feature_vector_scaled_two)[0]
        proba_two = second_model.predict_proba(feature_vector_scaled_two)[0]
        return {
            "isHeartSound": True,
            "isNormal": pred_two == 0,
            "confidence": proba_two.max() * 100
        }
    else:
        return {
            "isHeartSound": False,
            "confidence": proba_one.max() * 100
        }

# --- UI Components ---
def header():
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; background: linear-gradient(120deg, #1e293b, #334155); padding: 2rem; border-radius: 1rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: center; align-items: center; gap: 1rem;">
                <div style="font-size: 3rem; animation: pulse 2s infinite;">🩺</div>
                <h1 style="color: white; font-size: 2.5rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
                    Heart Sound Classifier
                </h1>
            </div>
            <p style="color: #94a3b8; margin-top: 1rem; font-size: 1.1rem;">
                Advanced AI-powered heart sound analysis
            </p>
        </div>
        <style>
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
        </style>
    """, unsafe_allow_html=True)

def result_card(title, message, status):
    colors = {
        "success": "#10b981",
        "error": "#ef4444",
        "warning": "#f59e0b"
    }
    bgcolor = {
        "success": "#d1fae5",
        "error": "#fee2e2",
        "warning": "#fef3c7"
    }
    icons = {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️"
    }
    
    st.markdown(f"""
        <div style="background: {bgcolor[status]}; 
                    padding: 1.5rem;
                    border-radius: 1rem;
                    margin: 1.5rem 0;
                    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
                    border: 1px solid {colors[status]};
                    transition: all 0.3s;">
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="font-size: 2rem;">{icons[status]}</div>
                <div>
                    <h3 style="color: {colors[status]}; 
                               margin: 0 0 0.5rem 0; 
                               font-size: 1.25rem;
                               font-weight: 600;">{title}</h3>
                    <p style="color: #374151; margin: 0;">{message}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def idle_state():
    st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="color: #1e293b; font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem;">
                Input Method
            </h2>
            <p style="color: #6b7280;">
                Choose how you want to input the heart sound
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize input_mode in session state if it doesn't exist
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "record"
    
    # Custom CSS for the radio buttons
    st.markdown("""
        <style>
            div.row-widget.stRadio > div {
                background-color: #f8fafc;
                padding: 0.5rem;
                border-radius: 1rem;
                display: flex;
                justify-content: center;
                gap: 1rem;
            }
            div.row-widget.stRadio > div [role="radiogroup"] {
                display: flex;
                justify-content: center;
                gap: 1rem;
            }
            div.row-widget.stRadio > div [role="radio"] {
                background-color: white;
                border: 2px solid #e2e8f0;
                padding: 0.75rem 2rem;
                border-radius: 0.75rem;
                transition: all 0.2s;
                cursor: pointer;
            }
            div.row-widget.stRadio > div [role="radio"]:hover {
                border-color: #3b82f6;
                transform: translateY(-2px);
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            }
            div.row-widget.stRadio > div [data-checked="true"] {
                background-color: #3b82f6;
                border-color: #3b82f6;
                color: white;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Create centered columns for the toggle
    _, col_mid, _ = st.columns([1, 2, 1])
    
    with col_mid:
        mode = st.radio(
            "Select Input Mode",
            ["Record", "Upload"],
            horizontal=True,
            label_visibility="collapsed",
            index=0 if st.session_state.input_mode == "record" else 1,
            key="mode_toggle"
        )
        st.session_state.input_mode = mode.lower()
    
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    
    # Custom CSS for buttons and upload area
    st.markdown("""
        <style>
            .stButton > button {
                width: 100%;
                padding: 1rem 2rem;
                font-size: 1.1rem;
                font-weight: 600;
                color: white;
                background: linear-gradient(45deg, #3b82f6, #2563eb);
                border: none;
                border-radius: 1rem;
                transition: all 0.3s;
                box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.2);
            }
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 8px -1px rgba(59, 130, 246, 0.3);
            }
            .stButton > button:active {
                transform: translateY(0);
            }
            .uploadFile {
                border: 2px dashed #3b82f6;
                border-radius: 1rem;
                padding: 2rem;
                text-align: center;
                background: #f0f9ff;
                transition: all 0.3s;
            }
            .uploadFile:hover {
                background: #e0f2fe;
                border-color: #2563eb;
            }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state.input_mode == "record":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style="text-align: center; margin-bottom: 1rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🎙️</div>
                    <p style="color: #6b7280; margin-bottom: 1rem;">Click below to start recording</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Start Recording", use_container_width=True):
                st.session_state.ui_state = "recording"
                st.rerun()
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div style="text-align: center; margin-bottom: 1rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
                    <p style="color: #6b7280; margin-bottom: 1rem;">Drag and drop your audio file here</p>
                </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Upload Audio File", 
                type=["wav", "mp3"],
                label_visibility="collapsed",
                key="audio_uploader"
            )
            if uploaded_file:
                audio_bytes = uploaded_file.read()
                y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
                st.session_state.audio = y
                st.session_state.sr = sr
                st.session_state.ui_state = "uploaded"
                st.rerun()

def recording_state():
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🎙️</div>
            <h2 style="color: #1e293b; font-size: 1.8rem; margin-bottom: 0.5rem;">Recording Audio</h2>
            <p style="color: #6b7280;">Set the duration and start recording</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Custom slider styling
    st.markdown("""
        <style>
            .stSlider {
                padding: 1rem 0;
            }
            .stSlider > div > div > div {
                background: linear-gradient(90deg, #3b82f6, #2563eb) !important;
            }
            .stSlider > div > div > div > div {
                background: white !important;
                border: 2px solid #3b82f6 !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        duration = st.slider(
            "Recording Duration (seconds)",
            min_value=3,
            max_value=10,
            value=st.session_state.duration,
            help="Slide to adjust recording duration"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎙️ Start Recording", use_container_width=True):
            with st.spinner("Recording in progress..."):
                # Add a progress animation
                st.markdown("""
                    <div style="display: flex; justify-content: center; margin: 1rem 0;">
                        <div style="width: 50px; height: 50px; background: #3b82f6; 
                                  border-radius: 50%; animation: pulse 1s infinite;">
                        </div>
                    </div>
                    <style>
                        @keyframes pulse {
                            0% { transform: scale(1); opacity: 1; }
                            50% { transform: scale(1.2); opacity: 0.5; }
                            100% { transform: scale(1); opacity: 1; }
                        }
                    </style>
                """, unsafe_allow_html=True)
                recording = sd.rec(int(duration * 22050), samplerate=22050, channels=1, dtype='float32')
                sd.wait()
                y = recording.flatten()
                st.session_state.audio = y
                st.session_state.sr = 22050
                st.session_state.ui_state = "uploaded"
                st.rerun()
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.session_state.ui_state = "idle"
            st.rerun()

def uploaded_state():
    st.markdown("<h2 style='text-align: center;'>Preview Recording</h2>", unsafe_allow_html=True)
    
    # Display audio player
    st.audio(st.session_state.audio, format="audio/wav", sample_rate=st.session_state.sr)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Over", use_container_width=True):
            st.session_state.ui_state = "idle"
            st.session_state.audio = None
            st.rerun()
    with col2:
        if st.button("Analyze", use_container_width=True):
            st.session_state.ui_state = "analyzing"
            st.rerun()

def analyzing_state():
    st.markdown("<h2 style='text-align: center;'>Analyzing...</h2>", unsafe_allow_html=True)
    
    with st.spinner("Processing audio..."):
        result = analyze_audio(st.session_state.audio, st.session_state.sr)
        st.session_state.result = result
        st.session_state.ui_state = "result"
        st.rerun()

def result_state():
    st.markdown("<h2 style='text-align: center;'>Analysis Results</h2>", unsafe_allow_html=True)
    
    result = st.session_state.result
    
    if result["isHeartSound"]:
        if result["isNormal"]:
            result_card(
                "Normal Heart Sound",
                "The analyzed sound appears to be a normal heart sound pattern.",
                "success"
            )
        else:
            result_card(
                "Abnormal Pattern Detected",
                "The heart sound shows potential abnormal characteristics. Please consult a healthcare professional.",
                "warning"
            )
    else:
        result_card(
            "Not a Heart Sound",
            "The audio was not recognized as a heart sound.",
            "error"
        )
    
    if st.button("Analyze Another Recording", use_container_width=True):
        st.session_state.ui_state = "idle"
        st.session_state.audio = None
        st.session_state.result = None
        st.rerun()

# --- Main UI Render ---
def main():
    header()
    
    if st.session_state.ui_state == "idle":
        idle_state()
    elif st.session_state.ui_state == "recording":
        recording_state()
    elif st.session_state.ui_state == "uploaded":
        uploaded_state()
    elif st.session_state.ui_state == "analyzing":
        analyzing_state()
    elif st.session_state.ui_state == "result":
        result_state()
    
    # Footer
    st.markdown("""
        <div style="text-align: center; margin-top: 2rem; color: #6b7280; font-size: 0.875rem;">
            Developed with ❤️ by Syed Asrar Zahoor
        </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
