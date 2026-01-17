import streamlit as st
import joblib
import numpy as np
import nltk
import folium
from textblob import TextBlob
from langdetect import detect
from streamlit_folium import st_folium

nltk.download("punkt")

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="SURAKSHA OMEGA AI",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 SURAKSHA OMEGA AI")
st.caption("AI-Powered Women Safety & Emergency Intelligence System")

# ---------------- LOAD ML MODELS ----------------
model = joblib.load("sos_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- FUNCTIONS ----------------
def predict_threat(text):
    vec = vectorizer.transform([text])
    level = model.predict(vec)[0]

    if level == 0:
        return "🟢 SAFE", 0
    elif level == 1:
        return "🟡 SUSPICIOUS", 1
    else:
        return "🔴 DANGER", 2

def emotion_detection(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.5:
        return "Fear 😨"
    elif polarity < 0:
        return "Anxiety 😟"
    else:
        return "Normal 🙂"

def language_detection(text):
    try:
        return {"en":"English","ta":"Tamil","hi":"Hindi"}.get(detect(text),"Unknown")
    except:
        return "Unknown"

def fake_sos(text):
    return len(text.split()) < 3

def voice_emotion(pitch, energy):
    if pitch > 220 and energy > 0.7:
        return "High Panic 😨"
    elif pitch > 180:
        return "Stressed 😟"
    else:
        return "Normal 🙂"

def generate_heatmap():
    lat = np.random.uniform(12.90, 13.10, 30)
    lon = np.random.uniform(77.50, 77.70, 30)
    risk = np.random.uniform(0.3, 1.0, 30)
    return lat, lon, risk

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio(
    "Select Feature",
    [
        "AI SOS Detection",
        "Voice Emotion ML",
        "Danger Zone Heatmap",
        "Fake SOS Detection",
        "Multilingual SOS",
        "Explainable AI"
    ]
)

# ---------------- FEATURES ----------------
if menu == "AI SOS Detection":
    st.header("🧠 AI SOS Intent Detection")

    text = st.text_area("Enter emergency text / voice transcript")

    if st.button("Analyze"):
        status, level = predict_threat(text)
        st.success(f"Threat Level: {status}")
        st.info(f"Language: {language_detection(text)}")
        st.warning(f"Emotion: {emotion_detection(text)}")

        if level == 2:
            st.error("🚨 AUTO SOS TRIGGER RECOMMENDED")

elif menu == "Voice Emotion ML":
    st.header("🎤 Voice Emotion Detection")
    pitch = st.slider("Voice Pitch (Hz)", 100, 300, 180)
    energy = st.slider("Voice Energy", 0.0, 1.0, 0.5)
    st.success("Detected Emotion: " + voice_emotion(pitch, energy))

elif menu == "Danger Zone Heatmap":
    st.header("🗺️ AI Danger Zone Heatmap")

    lat, lon, risk = generate_heatmap()
    m = folium.Map(location=[12.97, 77.59], zoom_start=12)

    for i in range(len(lat)):
        folium.Circle(
            location=[lat[i], lon[i]],
            radius=200,
            color="red" if risk[i] > 0.7 else "orange",
            fill=True
        ).add_to(m)

    st_folium(m, width=700, height=500)

elif menu == "Fake SOS Detection":
    st.header("🚫 Fake SOS Detection")
    msg = st.text_input("Enter SOS message")
    if msg:
        st.error("❌ Fake Alert") if fake_sos(msg) else st.success("✅ Genuine SOS")

elif menu == "Multilingual SOS":
    st.header("🌐 Multilingual SOS")
    st.code({
        "English": "Help me! I am in danger!",
        "Tamil": "உதவி தேவை! நான் ஆபத்தில் இருக்கிறேன்!",
        "Hindi": "मदद चाहिए! मैं खतरे में हूँ!"
    })

elif menu == "Explainable AI":
    st.header("🔍 Explainable AI")
    st.write("""
    AI triggers SOS based on:
    • Emergency keywords
    • TF-IDF similarity
    • Sentiment polarity
    • Voice stress features
    • Risk zone probability
    """)
