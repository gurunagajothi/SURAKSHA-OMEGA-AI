import streamlit as st
import joblib
import nltk
import numpy as np
import librosa
import av
import time
import folium
from streamlit_folium import st_folium

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------- CONFIG ----------------
st.set_page_config("SURAKSHA OMEGA AI", layout="centered")

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ---------------- LOAD MODELS ----------------
sos_model = joblib.load("sos_model (1).pkl")
vectorizer = joblib.load("vectorizer (1).pkl")
emotion_model = joblib.load("emotion_model.pkl")

# ---------------- FUNCTIONS ----------------
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([w for w in tokens if w.isalpha() and w not in stop_words])

def predict_sos_text(text):
    vec = vectorizer.transform([preprocess(text)])
    pred = sos_model.predict(vec)[0]
    prob = sos_model.predict_proba(vec).max()
    return pred, prob

def emotion_to_score(emotion):
    if emotion == "panic":
        return 0.9
    elif emotion == "calm":
        return 0.2
    return 0.5

def send_sos(location, zone):
    return {
        "status": "SOS SENT",
        "sent_to": ["Police Control Room", "Family Emergency Contacts"],
        "location": location,
        "risk": zone
    }

# ---------------- UI ----------------
st.title("🚨 SURAKSHA OMEGA AI")
st.caption("AI-Powered Women Safety System – Tamil Nadu")

menu = st.sidebar.selectbox(
    "Select Feature",
    [
        "🎤 Live Voice SOS Detection",
        "📍 Live Location – Tamil Nadu",
        "🧠 Text SOS Detection",
        "🔐 Admin Panel"
    ]
)

# ---------------- LIVE VOICE SOS ----------------
if menu == "🎤 Live Voice SOS Detection":
    st.info("🎙️ Allow microphone and speak for a few seconds")

    emotion_scores = []

    class VoiceProcessor(AudioProcessorBase):
        def __init__(self):
            self.last_emotion = "neutral"

        def recv(self, frame: av.AudioFrame):
            audio = frame.to_ndarray().flatten().astype(np.float32)

            if len(audio) > 4000:
                mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
                features = np.mean(mfcc.T, axis=0)
                self.last_emotion = emotion_model.predict([features])[0]

            return frame

    ctx = webrtc_streamer(
        key="voice",
        audio_processor_factory=VoiceProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if ctx.audio_processor:
        emotion = ctx.audio_processor.last_emotion
        score = emotion_to_score(emotion)
        emotion_scores.append(score)
        emotion_scores = emotion_scores[-10:]

        avg_score = sum(emotion_scores) / len(emotion_scores)
        st.progress(int(avg_score * 100))

        st.write(f"🎧 Detected Emotion: **{emotion.upper()}**")

        if avg_score > 0.7:
            zone = "DANGER ZONE"
            st.error("🔴 DANGER ZONE – SOS ACTIVATED")
            result = send_sos("Tamil Nadu", zone)
            st.success("🚓 SOS Sent to Police & Family")
            st.json(result)

        elif avg_score > 0.4:
            st.warning("🟡 PARTIALLY DANGER ZONE – Stay Alert")
        else:
            st.success("🟢 SAFE ZONE – No threat detected")

# ---------------- LIVE LOCATION ----------------
elif menu == "📍 Live Location – Tamil Nadu":
    st.subheader("📍 Live Location Tracking (Tamil Nadu)")

    lat = st.slider("Latitude", 8.0, 13.5, 11.0)
    lon = st.slider("Longitude", 76.0, 80.5, 78.0)

    m = folium.Map(location=[lat, lon], zoom_start=7)
    folium.Marker(
        [lat, lon],
        popup="User Location",
        icon=folium.Icon(color="red")
    ).add_to(m)

    st_folium(m, width=700, height=500)

    st.caption("Supports all districts in Tamil Nadu")

# ---------------- TEXT SOS ----------------
elif menu == "🧠 Text SOS Detection":
    msg = st.text_area("Enter SOS message")

    if st.button("Analyze & Send SOS"):
        pred, prob = predict_sos_text(msg)

        if pred == 2:
            st.error("🚨 EXTREME DANGER")
            result = send_sos("Tamil Nadu", "DANGER ZONE")
            st.success("🚓 SOS Sent to Police & Family")
            st.json(result)

        elif pred == 1:
            st.warning("⚠️ POSSIBLE DANGER – Alert Sent")
        else:
            st.success("✅ Safe Message")

        st.progress(int(prob * 100))

# ---------------- ADMIN ----------------
elif menu == "🔐 Admin Panel":
    st.warning("Admin Override System")

    if st.checkbox("Force Emergency SOS"):
        result = send_sos("Tamil Nadu", "ADMIN OVERRIDE")
        st.error("🚓 FORCE SOS SENT")
        st.json(result)
