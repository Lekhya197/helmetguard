import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import torch
import av
import threading
import numpy as np
import time
import requests

st.set_page_config(page_title="HelmetGuard AI - YOLOv5", layout="wide")

# Telegram Bot Config - replace with your actual bot token and chat ID
TELEGRAM_BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
TELEGRAM_CHAT_ID = '6674142283'

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")

@st.cache_resource(show_spinner=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

alert_audio_file = open("alert.mp3", "rb").read()

st.title("🎥 HelmetGuard AI - YOLOv5 Helmet Detection")

mode = st.sidebar.radio("Select Mode", ["Upload Video", "Webcam"])

def draw_boxes(frame, results):
    for *box, conf, cls in results.xyxy[0]:
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        conf_text = f"{label} {conf:.2f}"
        color = (0, 255, 0) if label == 'helmet_on' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, conf_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

alert_placeholder = st.empty()
audio_placeholder = st.empty()

if mode == "Upload Video":
    telegram_alert_sent = {"sent": False}  # Alert flag for upload video mode

    video_file = st.file_uploader("Upload a video for helmet detection", type=["mp4", "mov", "avi"])
    if video_file is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("❌ Could not open the uploaded video. Please try another file.")
        else:
            frame_placeholder = st.empty()
            helmet_metric = st.sidebar.empty()
            no_helmet_metric = st.sidebar.empty()
            alert_placeholder = st.sidebar.empty()
            audio_placeholder = st.empty()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.info("🎬 Video processing complete.")
                    telegram_alert_sent["sent"] = False  # reset flag after video ends
                    break
                results = model(frame)
                detections = results.xyxy[0]
                detections = detections[detections[:, 4] >= CONFIDENCE_THRESHOLD]

                labels = [model.names[int(cls)] for cls in detections[:, 5]]
                helmet_count = labels.count('helmet_on')
                no_helmet_count = labels.count('no_helmet')

                frame = draw_boxes(frame, results)

                helmet_metric.metric("✅ Helmet On", helmet_count)
                no_helmet_metric.metric("🚨 No Helmet", no_helmet_count)

                if no_helmet_count > 0:
                    alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                    audio_placeholder.audio(alert_audio_file, format="audio/mp3", start_time=0)

                    if not telegram_alert_sent["sent"]:
                        send_telegram_alert("🚨 Alert: Riders without helmets detected in uploaded video by HelmetGuard AI!")
                        telegram_alert_sent["sent"] = True
                else:
                    alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                    audio_placeholder.empty()
                    telegram_alert_sent["sent"] = False

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")

                time.sleep(0.03)
            cap.release()
    else:
        st.info("⬆️ Please upload a video to begin helmet detection.")

else:  # Webcam mode

    alert_state = {"no_helmet": False}
    telegram_alert_sent = {"sent": False}

    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            labels = [model.names[int(cls)] for cls in results.xyxy[0][:, 5]]
            alert_state["no_helmet"] = labels.count('no_helmet') > 0
            img = draw_boxes(img, results)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(key="helmet-detection", video_processor_factory=VideoProcessor,
                                 media_stream_constraints={"video": True, "audio": False},
                                 async_processing=True)

    def update_ui():
        while True:
            if webrtc_ctx.state.playing:
                if alert_state["no_helmet"]:
                    alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                    audio_placeholder.audio(alert_audio_file, format="audio/mp3", start_time=0)

                    if not telegram_alert_sent["sent"]:
                        send_telegram_alert("🚨 Alert: Riders without helmets detected by HelmetGuard AI!")
                        telegram_alert_sent["sent"] = True
                else:
                    alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                    audio_placeholder.empty()
                    telegram_alert_sent["sent"] = False
            else:
                alert_placeholder.info("📷 Webcam inactive.")
                audio_placeholder.empty()
                telegram_alert_sent["sent"] = False
            time.sleep(0.5)

    thread = threading.Thread(target=update_ui, daemon=True)
    thread.start()
