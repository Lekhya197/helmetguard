import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch
import cv2
import av
import time
import threading
import numpy as np
import pandas as pd
import requests
import os
from queue import Queue

# Page configuration
st.set_page_config(page_title="HelmetGuard AI - YOLOv5", page_icon="🛡️", layout="wide")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

# Telegram Config
TELEGRAM_BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
TELEGRAM_CHAT_ID = '6674142283'

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        r = requests.post(url, data=payload, timeout=5)
        return r.status_code == 200
    except Exception as e:
        st.error(f"Telegram Error: {e}")
        return False

# Draw detection boxes
def draw_boxes(frame, df):
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = f"{row['name']} {row['confidence']:.2f}"
        color = (0, 255, 0) if row['name'] == 'helmet_on' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration")
    conf_helmet = st.slider("Helmet Confidence", 0.0, 1.0, 0.4, 0.01)
    conf_no_helmet = st.slider("No Helmet Confidence", 0.0, 1.0, 0.5, 0.01)
    mode = st.radio("Select Mode", ["Upload Video", "Webcam"])
    st.subheader("📊 Detection Metrics")
    col1, col2 = st.columns(2)
    helmet_metric = col1.empty()
    no_helmet_metric = col2.empty()
    status_message = st.empty()
    alert_placeholder = st.empty()

# Session state setup
st.session_state.setdefault('alert_sent', False)
st.session_state.setdefault('last_alert_time', 0)

st.title("🛡️ HelmetGuard AI - YOLOv5 Helmet Detection")

# === VIDEO UPLOAD MODE ===
if mode == "Upload Video":
    st.subheader("📁 Upload a Video")
    video_file = st.file_uploader("Choose a file", type=["mp4", "avi", "mov", "mkv"])

    if video_file:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 0.03
        queue = Queue()

        def process_frames():
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    queue.put(("end", None, None, None))
                    break

                results = model(frame)
                df = results.pandas().xyxy[0]
                df_helmet = df[(df['name'] == 'helmet_on') & (df['confidence'] >= conf_helmet)]
                df_no_helmet = df[(df['name'] == 'no_helmet') & (df['confidence'] >= conf_no_helmet)]

                frame = draw_boxes(frame, pd.concat([df_helmet, df_no_helmet]))
                queue.put(("frame", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), len(df_helmet), len(df_no_helmet)))
                time.sleep(delay)

            cap.release()
            os.remove(temp_path)

        threading.Thread(target=process_frames, daemon=True).start()
        placeholder = st.empty()

        while True:
            if not queue.empty():
                msg, frame, h_count, nh_count = queue.get()
                if msg == "end":
                    status_message.success("✅ Processing Complete")
                    break
                placeholder.image(frame, channels="RGB")
                helmet_metric.metric("✅ Helmet On", h_count)
                no_helmet_metric.metric("🚨 No Helmet", nh_count)
                if nh_count > 0:
                    status_message.error("❌ No Helmet Detected")
                    alert_placeholder.error("⚠️ Alert Triggered")
                    if not st.session_state.alert_sent:
                        if send_telegram_alert(f"⚠️ ALERT: {nh_count} rider(s) without helmet detected!"):
                            st.session_state.alert_sent = True
                else:
                    status_message.success("✅ All Safe")
                    st.session_state.alert_sent = False
            time.sleep(0.01)

    else:
        st.info("⬆️ Upload a video to begin")

# === WEBCAM MODE ===
else:
    st.subheader("🎥 Real-time Webcam Detection")

    class VideoProcessor:
        def __init__(self):
            self.h_count = 0
            self.nh_count = 0
            self.p_count = 0
            self.trigger = False

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            df = results.pandas().xyxy[0]

            df_helmet = df[(df['name'] == 'helmet_on') & (df['confidence'] >= conf_helmet)]
            df_no_helmet = df[(df['name'] == 'no_helmet') & (df['confidence'] >= conf_no_helmet)]
            df_person = df[df['name'] == 'person']

            self.h_count = len(df_helmet)
            self.nh_count = len(df_no_helmet)
            self.p_count = len(df_person)
            self.trigger = self.nh_count > 0 or self.p_count > 0

            img = draw_boxes(img, pd.concat([df_helmet, df_no_helmet, df_person]))
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="realtime",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    def webcam_loop():
        while True:
            if ctx.state.playing and ctx.video_processor:
                proc = ctx.video_processor
                helmet_metric.metric("✅ Helmet On", proc.h_count)
                no_helmet_metric.metric("🚨 No Helmet", proc.nh_count)

                if proc.nh_count > 0:
                    status_message.error("❌ No Helmet Detected")
                    msg = f"⚠️ {proc.nh_count} no-helmet; 👤 {proc.p_count} person(s)"
                    alert_placeholder.error(msg)
                    if not st.session_state.alert_sent or time.time() - st.session_state.last_alert_time > 60:
                        if send_telegram_alert(msg):
                            st.session_state.alert_sent = True
                            st.session_state.last_alert_time = time.time()
                else:
                    status_message.success("✅ All Safe")
                    st.session_state.alert_sent = False
            else:
                helmet_metric.empty()
                no_helmet_metric.empty()
                status_message.info("📷 Waiting for webcam...")
            time.sleep(0.1)

    threading.Thread(target=webcam_loop, daemon=True).start()
