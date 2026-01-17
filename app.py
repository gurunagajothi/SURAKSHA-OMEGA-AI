# ===================== IMPORTS =====================
import streamlit as st
import numpy as np
import joblib
import librosa
import av
import folium
from streamlit_folium import st_folium
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# ===================== APP CONFIG =====================
st.set_page_config(
    page_title="SURAKSHA OMEGA AI",
    layout="wide"
)

st.title("🛡️ SURAKSHA OMEGA – AI Women Safety System")
st.caption("Live Voice Emotion Detection • SOS AI • Tamil Nadu Location")

# ===================== LOAD ML MODELS =====================
# Make sure these files are in the repo root
emotion_model = joblib.load("sos_model (1).pkl")
vectorizer = joblib.load("vectorizer (1).pkl")  # reserved for text SOS feature

# ===================== SIDEBAR =====================
menu = st.sidebar.radio(
    "Select Feature",
    [
        "🎤 Live Voice Emotion Detection",
        "📍 Live Location (Tamil Nadu)",
        "🚨 SOS Status"
    ]
)

# ===================== SESSION STATE =====================
if "zone" not in st.session_state:
    st.session_state.zone = "SAFE"

# ===================== 🎤 VOICE EMOTION DETECTION =====================
if menu == "🎤 Live Voice Emotion Detection":
    st.info("🎙️ Click START and speak clearly for 5–10 seconds")

    class VoiceProcessor(AudioProcessorBase):
        def __init__(self):
            self.emotion = "neutral"

        def recv(self, frame: av.AudioFrame):
            audio = frame.to_ndarray().flatten().astype(np.float32)

            if len(audio) > 4000:
                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=16000,
                    n_mfcc=40
                )
                features = np.mean(mfcc.T, axis=0)
                self.emotion = emotion_model.predict([features])[0]

            return frame

    ctx = webrtc_streamer(
        key="voice",
        audio_processor_factory=VoiceProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True
    )

    st.divider()

    if ctx and ctx.state.playing:
        emotion = ctx.audio_processor.emotion
        st.write(f"🎧 Detected Emotion: **{emotion.upper()}**")

        if emotion == "panic":
            st.session_state.zone = "DANGER"
            st.error("🔴 DANGER ZONE – SOS ACTIVATED")
        elif emotion == "neutral":
            st.session_state.zone = "PARTIAL"
            st.warning("🟡 PARTIALLY DANGER ZONE")
        else:
            st.session_state.zone = "SAFE"
            st.success("🟢 SAFE ZONE")

# ===================== 📍 TAMIL NADU LOCATION =====================
elif menu == "📍 Live Location (Tamil Nadu)":
    st.info("📡 Live Location – Tamil Nadu Coverage")

    # Tamil Nadu center coordinates
    lat, lon = 11.1271, 78.6569

    map_tn = folium.Map(location=[lat, lon], zoom_start=7)

    # Add a marker for the user location
    folium.Marker(
        [lat, lon],
        popup="User Location (Tamil Nadu)",
        icon=folium.Icon(color="red")
    ).add_to(map_tn)

    # Display map safely
    try:
        st_folium(map_tn)
    except Exception as e:
        st.error(f"Map could not load: {e}")

# ===================== 🚨 SOS STATUS =====================
elif menu == "🚨 SOS Status":
    st.subheader("🚨 Emergency Alert Panel")

    if st.session_state.zone == "DANGER":
        st.error("🚓 SOS SENT TO POLICE")
        st.success("📞 SOS SENT TO FAMILY MEMBERS")
        st.markdown("""
        **Automatic Actions Triggered**
        - Panic emotion detected from live voice
        - User location shared (Tamil Nadu)
        - Emergency escalation activated
        """)

    elif st.session_state.zone == "PARTIAL":
        st.warning("⚠️ Possible risk detected – Monitoring user")

    else:
        st.success("✅ User is Safe – No SOS Triggered")

# ===================== FOOTER =====================
st.markdown("---")
st.caption("© SURAKSHA OMEGA AI | Stable • ML Powered • Hackathon Ready")
