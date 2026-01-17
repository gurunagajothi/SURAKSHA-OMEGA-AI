import streamlit as st
import joblib
import nltk
import numpy as np
import librosa
import av

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

# ---------------- NLP FUNCTIONS ----------------
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([w for w in tokens if w.isalpha() and w not in stop_words])

def predict_sos(text):
    vec = vectorizer.transform([preprocess(text)])
    pred = sos_model.predict(vec)[0]
    prob = sos_model.predict_proba(vec).max()
    return pred, prob

# ---------------- AUDIO FEATURE EXTRACTION ----------------
def extract_audio_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

# ---------------- AUDIO PROCESSOR ----------------
class EmotionAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.emotion = "Listening..."

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray().flatten().astype(np.float32)

        if len(audio) > 4000:
            features = extract_audio_features(audio)
            prediction = emotion_model.predict([features])[0]
            self.emotion = prediction

        return frame

# ---------------- UI ----------------
st.title("🚨 SURAKSHA OMEGA AI")
st.caption("Live AI-Powered Women Safety System")

menu = st.sidebar.selectbox(
    "Select Feature",
    [
        "SOS Detection (Text ML)",
        "Fake SOS Detection",
        "🎤 Live Voice Emotion Detection",
        "Maps Heatmap",
        "Admin Panel"
    ]
)

# ---------------- TEXT SOS ----------------
if menu == "SOS Detection (Text ML)":
    msg = st.text_area("Enter SOS message")

    if st.button("Analyze"):
        pred, prob = predict_sos(msg)

        if pred == 2:
            st.error("🚨 EXTREME DANGER – AUTO SOS")
        elif pred == 1:
            st.warning("⚠️ POSSIBLE THREAT")
        else:
            st.success("✅ SAFE MESSAGE")

        st.progress(int(prob * 100))
        st.caption(f"Confidence: {round(prob*100,2)}%")

# ---------------- FAKE SOS ----------------
elif menu == "Fake SOS Detection":
    msg = st.text_input("Enter SOS message")

    if msg:
        if len(msg.split()) < 3:
            st.error("❌ Fake Alert Detected")
        else:
            st.success("✅ Genuine SOS")

# ---------------- LIVE VOICE EMOTION ----------------
elif menu == "🎤 Live Voice Emotion Detection":
    st.info("🎙️ Allow microphone access and speak")

    ctx = webrtc_streamer(
        key="emotion",
        audio_processor_factory=EmotionAudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if ctx.audio_processor:
        emotion = ctx.audio_processor.emotion

        st.subheader(f"Detected Emotion: **{emotion.upper()}**")

        if emotion == "panic":
            st.error("😱 PANIC DETECTED – AUTO SOS ACTIVATED")
        elif emotion == "calm":
            st.success("🙂 Calm voice detected")

# ---------------- MAPS HEATMAP ----------------
elif menu == "Maps Heatmap":
    hour = st.slider("Time (24h)", 0, 23, 22)
    incidents = st.slider("Past Incidents", 0, 10, 6)

    score = 0.3
    if hour >= 20:
        score += 0.3
    if incidents > 5:
        score += 0.4

    score = min(score, 1.0)

    if score > 0.7:
        st.error("🔴 HIGH RISK ZONE")
    elif score > 0.4:
        st.warning("🟡 MODERATE RISK")
    else:
        st.success("🟢 SAFE ZONE")

    st.progress(int(score * 100))

# ---------------- ADMIN PANEL ----------------
elif menu == "Admin Panel":
    st.warning("🔐 Admin Override Panel")

    if st.checkbox("Force SOS"):
        st.error("🚓 SOS SENT TO AUTHORITIES")
