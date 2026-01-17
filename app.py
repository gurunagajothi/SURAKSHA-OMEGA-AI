import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import folium
import io
import base64
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from streamlit_folium import st_folium
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av

# -------------------- APP CONFIG --------------------
st.set_page_config(page_title="SURAKSHA OMEGA AI", layout="wide")
st.title("🛡️ SURAKSHA OMEGA – Women Safety System")
st.caption("Live Voice Risk Detection + SOS + Tamil Nadu Map")

# -------------------- TRAIN SAMPLE SAFETY MODEL --------------------
@st.cache_data
def train_model():
    np.random.seed(42)
    n = 2000
    lat = np.random.normal(13.08, 0.03, n)
    lon = np.random.normal(80.27, 0.03, n)
    hour = np.random.randint(0, 24, n)
    audio_energy = np.random.uniform(0, 1, n)

    risk = []
    for h, e in zip(hour, audio_energy):
        score = 1
        if h > 20 or h < 6: score -= 0.4
        if e > 0.6: score -= 0.4
        risk.append(1 if score > 0.4 else 0)

    X = pd.DataFrame({"lat": lat, "lon": lon, "hour": hour, "energy": audio_energy})
    y = np.array(risk)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)
    return model

model = train_model()

# -------------------- AUDIO FEATURE EXTRACTION --------------------
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    energy = np.mean(librosa.feature.rms(y=y))
    return energy, y, sr

# -------------------- SAFETY ANALYSIS --------------------
def analyze_safety(lat, lon, hour, audio_path):
    if audio_path is None:
        return "❌ Please record voice first", None, None

    energy, y, sr = extract_audio_features(audio_path)
    features = [[lat, lon, hour, energy]]
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][prediction] * 100
    status = "🚨 HIGH RISK DETECTED" if prediction == 0 else "✅ SAFE ZONE"

    # Waveform plot
    plt.figure(figsize=(10,3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Live Voice Waveform")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_data = base64.b64encode(buf.read()).decode()
    plt.close()

    # Folium map
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], popup=status).add_to(m)

    return (
        f"""
### 🛡️ Safety Result
**Status:** {status}
**Confidence:** {confidence:.2f}%
**Audio Energy:** {energy:.3f}
""",
        img_data,
        m
    )

# -------------------- STREAMLIT UI --------------------
st.sidebar.header("Input Parameters")
lat = st.sidebar.number_input("Latitude", value=13.082)
lon = st.sidebar.number_input("Longitude", value=80.27)
hour = st.sidebar.slider("Hour", 0, 23, value=datetime.now().hour)

st.header("🎤 Record Live Voice")
audio_file = st.file_uploader("Upload or Record Voice (WAV/MP3)", type=["wav","mp3"])

if st.button("🔍 Analyze Safety"):
    report, waveform_img, map_obj = analyze_safety(lat, lon, hour, audio_file)
    st.markdown(report)
    if waveform_img:
        st.image(base64.b64decode(waveform_img))
    if map_obj:
        try:
            st_folium(map_obj)
        except Exception as e:
            st.error(f"Map could not load: {e}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("© SURAKSHA OMEGA AI | Live Voice Safety + SOS + TN Map")
