import streamlit as st
import cv2
import numpy as np
import time
import torch
import requests
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Telegram Bot Credentials
BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
CHAT_ID = '6674142283'
COOLDOWN_SECONDS = 30
last_alert_time = 0

# Streamlit Setup
st.set_page_config(page_title="HelmetGuard AI - YOLOv5", layout="wide")
st.title("🪖 HelmetGuard AI - Helmet Detection")
st.sidebar.header("Settings")

# Load model with caching
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    model.conf = 0.3
    return model

model = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Confirm required classes are in the model
required_classes = {"helmet_on", "no_helmet"}
model_classes = set(model.names.values())
missing_classes = required_classes - model_classes
if missing_classes:
    st.error(f"❌ Required classes not found in model: {missing_classes}")
    st.stop()

# Telegram alert function
def send_telegram_alert(message):
    global last_alert_time
    if time.time() - last_alert_time < COOLDOWN_SECONDS:
        return
    last_alert_time = time.time()
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        st.error(f"Telegram error: {e}")

# Sidebar options
mode = st.sidebar.selectbox("Select Mode", ["Upload Image/Video", "Webcam Detection"])
confidence = st.sidebar.slider("Confidence Threshold", 0.3, 1.0, 0.4, 0.05)

# Alerts
alert_audio_file = "alert.mp3"
audio_placeholder = st.empty()

# Upload Mode
if mode == "Upload Image/Video":
    uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        file_type = uploaded_file.type
        np_arr = np.frombuffer(file_bytes, np.uint8)

        if "image" in file_type:
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model(image_rgb)
            df = results.pandas().xyxy[0]

            helmet_count = (df['name'] == 'helmet_on').sum()
            no_helmet_count = (df['name'] == 'no_helmet').sum()

            st.image(results.render()[0], caption="Processed Image", use_column_width=True)
            st.sidebar.metric("✅ Helmet On", helmet_count)
            st.sidebar.metric("🚨 No Helmet", no_helmet_count)

            if no_helmet_count > 0:
                send_telegram_alert("🚨 Helmet violation detected in uploaded image!")
                audio_placeholder.audio(alert_audio_file, format="audio/mp3")

        elif "video" in file_type:
            tfile = open("temp_video.mp4", "wb")
            tfile.write(file_bytes)
            cap = cv2.VideoCapture("temp_video.mp4")

            output_frames = []
            helmet_count, no_helmet_count = 0, 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)
                df = results.pandas().xyxy[0]

                helmet_count += (df['name'] == 'helmet_on').sum()
                no_helmet_count += (df['name'] == 'no_helmet').sum()

                output_frames.append(results.render()[0])

            cap.release()
            st.image(output_frames, caption="Processed Frames", use_column_width=True)
            st.sidebar.metric("✅ Helmet On", helmet_count)
            st.sidebar.metric("🚨 No Helmet", no_helmet_count)

            if no_helmet_count > 0:
                send_telegram_alert("🚨 Helmet violation detected in uploaded video!")
                audio_placeholder.audio(alert_audio_file, format="audio/mp3")

# Webcam Detection Mode
elif mode == "Webcam Detection":
    stframe = st.empty()
    helmet_sidebar = st.sidebar.empty()
    no_helmet_sidebar = st.sidebar.empty()

    cap = cv2.VideoCapture(0)
    st.warning("🔴 Press 'Stop' to end webcam detection.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        df = results.pandas().xyxy[0]

        helmet_count = (df['name'] == 'helmet_on').sum()
        no_helmet_count = (df['name'] == 'no_helmet').sum()

        stframe.image(results.render()[0], channels="BGR", use_column_width=True)
        helmet_sidebar.metric("✅ Helmet On", int(helmet_count))
        no_helmet_sidebar.metric("🚨 No Helmet", int(no_helmet_count))

        if no_helmet_count > 0:
            send_telegram_alert("🚨 Helmet violation detected via webcam!")
            audio_placeholder.audio(alert_audio_file, format="audio/mp3")

        if not st.session_state.get("webcam_running", True):
            break

    cap.release()
    st.success("✅ Webcam stopped.")
