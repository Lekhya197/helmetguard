import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import torch
import cv2
import av
import numpy as np
import time
import threading
import pygame
import requests

# Telegram Bot Info
BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
CHAT_ID = '6674142283'

# Streamlit Config
st.set_page_config(page_title="HelmetGuard AI - YOLOv5", layout="wide")
st.title("🪖 HelmetGuard AI - Helmet Detection with YOLOv5")

# Load alert audio file bytes for Streamlit audio player
alert_audio_file = open("alert.mp3", "rb").read()

# Initialize pygame mixer once for alert sound playback in webcam mode
pygame.mixer.init()

# Load YOLOv5 model with cache
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

# Sidebar configuration
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
alert_threshold = 3  # Number of consecutive frames with no_helmet to trigger alert
cooldown_period = 30  # seconds cooldown between alerts

# Alert Manager with cooldown for webcam mode
class CooldownAlertManager:
    def __init__(self, cooldown_period=30):
        self.last_alert_time = 0
        self.cooldown_period = cooldown_period

    def can_alert(self):
        current_time = time.time()
        if current_time - self.last_alert_time >= self.cooldown_period:
            self.last_alert_time = current_time
            return True
        return False

# For upload video mode: track if helmet was detected previously to avoid repeated alerts
class UploadAlertManager:
    def __init__(self):
        self.triggered = False
        self.helmet_previously_detected = False

    def should_alert(self, labels):
        no_helmet = 'no_helmet' in labels
        helmet_on = 'helmet_on' in labels

        if helmet_on:
            self.helmet_previously_detected = True

        if not no_helmet:
            self.triggered = False
            return False

        if no_helmet and self.helmet_previously_detected and not self.triggered:
            self.triggered = True
            self.helmet_previously_detected = False
            return True

        return False

upload_alert_manager = UploadAlertManager()

# Function to send Telegram message asynchronously
def send_telegram_message_async(message):
    def send():
        try:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          data={'chat_id': CHAT_ID, 'text': message})
        except Exception as e:
            st.warning(f"Telegram error: {e}")
    threading.Thread(target=send, daemon=True).start()

# Play alert sound for webcam mode (pygame mixer already initialized)
def play_alert_sound():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load("alert.mp3")
        pygame.mixer.music.play()

# Draw bounding boxes on frames
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

mode = st.sidebar.radio("Select Mode", ["Upload Video", "Webcam"])

if mode == "Upload Video":
    video_file = st.file_uploader("Upload a video for helmet detection", type=["mp4", "mov", "avi"])
    if video_file is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            st.error("❌ Could not open the uploaded video.")
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

                if upload_alert_manager.should_alert(labels):
                    alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                    audio_placeholder.audio(alert_audio_file, format="audio/mp3", start_time=0)
                    send_telegram_message_async("🚨 Helmet violation detected in uploaded video!")
                else:
                    alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                    audio_placeholder.empty()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
                time.sleep(0.03)

            cap.release()
    else:
        st.info("⬆️ Please upload a video to begin helmet detection.")

else:  # Webcam mode

    alert_manager = CooldownAlertManager(cooldown_period=cooldown_period)
    alert_state = {
        "no_helmet_count": 0
    }

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img_bgr = frame.to_ndarray(format="bgr24")
            img_bgr = cv2.resize(img_bgr, (1280, 720))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            results = model(img_rgb)
            df = results.pandas().xyxy[0]

            filtered = df[
                (df['confidence'] >= CONFIDENCE_THRESHOLD)
            ]

            labels = list(filtered['name'])

            if 'no_helmet' in labels:
                alert_state["no_helmet_count"] += 1
            else:
                alert_state["no_helmet_count"] = 0

            if alert_state["no_helmet_count"] >= alert_threshold:
                if alert_manager.can_alert():
                    send_telegram_message_async("🚨 Helmet violation detected in webcam feed!")
                    play_alert_sound()

            for _, row in filtered.iterrows():
                xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                label = f"{row['name']} {row['confidence']:.2f}"
                color = (0, 255, 0) if row['name'] == 'helmet_on' else (0, 0, 255)
                cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img_bgr, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    alert_placeholder = st.empty()
    audio_placeholder = st.empty()

    ctx = webrtc_streamer(
        key="helmet-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    def update_ui():
        while True:
            if ctx.state.playing:
                if alert_state["no_helmet_count"] >= alert_threshold:
                    alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                    audio_placeholder.audio(alert_audio_file, format="audio/mp3", start_time=0)
                else:
                    alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                    audio_placeholder.empty()
            else:
                alert_placeholder.info("📷 Webcam inactive.")
                audio_placeholder.empty()
            time.sleep(0.5)

    threading.Thread(target=update_ui, daemon=True).start()
