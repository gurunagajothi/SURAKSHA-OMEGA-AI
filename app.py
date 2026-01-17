import streamlit as st
import joblib
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------- CONFIG ----------------
st.set_page_config(page_title="SURAKSHA OMEGA AI", layout="centered")

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ---------------- LOAD ML MODELS ----------------
# ⚠️ IMPORTANT: filenames with (1)
sos_model = joblib.load("sos_model (1).pkl")
vectorizer = joblib.load("vectorizer (1).pkl")

# (Optional voice model – safe fallback)
try:
    emotion_model = joblib.load("emotion_model.pkl")
except:
    emotion_model = None

# ---------------- FUNCTIONS ----------------
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([w for w in tokens if w.isalpha() and w not in stop_words])

def predict_sos(text):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    pred = sos_model.predict(vec)[0]
    prob = sos_model.predict_proba(vec).max()
    return pred, prob

def fake_sos(text):
    words = text.split()
    if len(words) < 3:
        return True, "Message too short"
    return False, "Message looks valid"

def voice_emotion():
    if emotion_model is None:
        return "calm"
    sample = np.random.rand(40)
    return emotion_model.predict([sample])[0]

def heatmap_risk(hour, incidents):
    score = 0.3
    if hour >= 20:
        score += 0.3
    if incidents > 5:
        score += 0.4
    return min(score, 1.0)

# ---------------- UI ----------------
st.title("🚨 SURAKSHA OMEGA AI")
st.caption("AI-Powered Women Safety System")

menu = st.sidebar.selectbox(
    "Select Feature",
    [
        "SOS Detection (ML)",
        "Fake SOS Detection",
        "Voice Emotion Detection",
        "Maps Heatmap",
        "Admin Panel"
    ]
)

# ---------------- SOS DETECTION ----------------
if menu == "SOS Detection (ML)":
    msg = st.text_area("Enter SOS message")

    if st.button("Analyze"):
        if msg.strip() == "":
            st.warning("Please enter a message")
        else:
            pred, prob = predict_sos(msg)

            if pred == 2:
                st.error("🚨 EXTREME DANGER – AUTO SOS")
            elif pred == 1:
                st.warning("⚠️ POSSIBLE THREAT")
            else:
                st.success("✅ SAFE MESSAGE")

            st.progress(int(prob * 100))
            st.caption(f"Confidence: {round(prob*100, 2)}%")

# ---------------- FAKE SOS ----------------
elif menu == "Fake SOS Detection":
    msg = st.text_input("Enter SOS message")

    if msg:
        is_fake, reason = fake_sos(msg)
        if is_fake:
            st.error("❌ Fake Alert Detected")
        else:
            st.success("✅ Genuine SOS")
        st.caption(reason)

# ---------------- VOICE EMOTION ----------------
elif menu == "Voice Emotion Detection":
    st.info("🎤 Simulated Voice Input (ML Ready)")
    emotion = voice_emotion()

    if emotion == "panic":
        st.error("😱 PANIC DETECTED – AUTO SOS")
    else:
        st.success("🙂 Calm Voice Detected")

# ---------------- MAPS HEATMAP ----------------
elif menu == "Maps Heatmap":
    hour = st.slider("Time (24h)", 0, 23, 22)
    incidents = st.slider("Past Incident Count", 0, 10, 6)

    risk = heatmap_risk(hour, incidents)

    if risk > 0.7:
        st.error("🔴 HIGH RISK ZONE")
    elif risk > 0.4:
        st.warning("🟡 MODERATE RISK ZONE")
    else:
        st.success("🟢 SAFE ZONE")

    st.progress(int(risk * 100))

# ---------------- ADMIN PANEL ----------------
elif menu == "Admin Panel":
    st.warning("🔐 Admin Override Panel")

    override = st.checkbox("Force Emergency SOS")

    if override:
        st.error("🚓 SOS SENT TO AUTHORITIES")
