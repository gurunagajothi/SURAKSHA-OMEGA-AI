import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -------------------------------
# Setup
# -------------------------------
st.set_page_config(page_title="SURAKSHA OMEGA AI", layout="centered")

nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# -------------------------------
# Load ML Models
# -------------------------------
model = joblib.load("sos_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------------------
# Preprocess
# -------------------------------
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([w for w in tokens if w.isalpha() and w not in stop_words])

# -------------------------------
# SOS Prediction
# -------------------------------
def predict_sos(text):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return pred, prob

# -------------------------------
# Fake SOS Detection (FIXED)
# -------------------------------
def fake_sos(text):
    words = text.split()

    if len(words) < 3:
        return True, "Message too short"

    if all(w.lower() in ["help", "save", "me"] for w in words):
        return True, "Repeated generic words"

    return False, "Message has context"

# -------------------------------
# UI
# -------------------------------
st.title("🚨 SURAKSHA OMEGA AI")
st.caption("AI-Powered Women Safety System")

menu = st.sidebar.radio(
    "Select Feature",
    [
        "SOS Intent Detection",
        "Fake SOS Detection",
        "Threat Level",
        "Language Switch"
    ]
)

# -------------------------------
# SOS INTENT
# -------------------------------
if menu == "SOS Intent Detection":
    st.header("🧠 SOS Intent Detection (ML)")

    msg = st.text_area("Enter message")

    if st.button("Analyze"):
        if msg:
            pred, prob = predict_sos(msg)

            if pred == 2:
                st.error("🚨 EXTREME DANGER – AUTO SOS")
            elif pred == 1:
                st.warning("⚠️ POSSIBLE DANGER")
            else:
                st.success("✅ SAFE MESSAGE")

            st.progress(int(prob * 100))
            st.caption(f"Confidence: {round(prob*100,2)}%")

# -------------------------------
# FAKE SOS (ERROR FIXED)
# -------------------------------
elif menu == "Fake SOS Detection":
    st.header("🚫 Fake SOS Detection")

    msg = st.text_input("Enter SOS message")

    if msg:
        is_fake, reason = fake_sos(msg)

        if is_fake:
            st.error("❌ Fake Alert Detected")
            st.caption(f"Reason: {reason}")
        else:
            st.success("✅ Genuine SOS")
            st.caption("Message appears valid")

# -------------------------------
# THREAT LEVEL
# -------------------------------
elif menu == "Threat Level":
    st.header("📊 Threat Level Estimation")

    time = st.slider("Time (24h)", 0, 23, 21)
    area = st.selectbox("Area Type", ["crowded", "isolated"])

    score = 0.4
    if time >= 20:
        score += 0.3
    if area == "isolated":
        score += 0.3

    score = min(score, 1.0)

    if score > 0.7:
        st.error("🔴 HIGH RISK AREA")
    elif score > 0.4:
        st.warning("🟡 MODERATE RISK")
    else:
        st.success("🟢 SAFE AREA")

    st.progress(int(score * 100))

# -------------------------------
# LANGUAGE SWITCH
# -------------------------------
elif menu == "Language Switch":
    st.header("🌐 Language Support")

    lang = st.selectbox("Select Language", ["English", "Tamil", "Hindi"])

    if lang == "Tamil":
        st.success("அவசர SOS செயல்படுத்தப்பட்டது")
    elif lang == "Hindi":
        st.success("आपातकालीन SOS सक्रिय")
    else:
        st.success("Emergency SOS Activated")
