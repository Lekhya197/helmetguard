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

st.set_page_config(page_title="HelmetGuard AI YOLOv5", layout="wide")

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

st.title("🎥 HelmetGuard AI - YOLOv5 Helmet Detection")

def draw_boxes(frame, results_df):
    for _, row in results_df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = f"{row['name']} {row['confidence']:.2f}"
        color = (0, 255, 0) if row['name'] == 'helmet_on' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# Telegram Bot Config
TELEGRAM_BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
TELEGRAM_CHAT_ID = '6674142283'

def send_telegram_alert(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print("Telegram alert failed:", e)
        return False

mode = st.sidebar.radio("Select Mode", ["Upload Video", "Webcam"])

alert_placeholder = st.sidebar.empty()
helmet_metric = st.sidebar.empty()
no_helmet_metric = st.sidebar.empty()

# State to avoid spamming telegram alerts
alert_sent_state = {"sent": False}

if mode == "Upload Video":

    video_file = st.file_uploader("Upload a video for helmet detection", type=["mp4", "mov", "avi"])

    if video_file is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        cap = cv2.VideoCapture(temp_video_path)

        if not cap.isOpened():
            st.error("❌ Could not open the uploaded video. Please try another file.")
        else:
            st.success("✅ Processing video...")
            frame_placeholder = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.info("🎬 Video processing complete.")
                    helmet_metric.empty()
                    no_helmet_metric.empty()
                    alert_placeholder.empty()
                    alert_sent_state["sent"] = False
                    break

                results = model(frame)
                df = results.pandas().xyxy[0]

                helmet_count = (df['name'] == 'helmet_on').sum()
                no_helmet_count = (df['name'] == 'no_helmet').sum()

                frame = draw_boxes(frame, df)

                if helmet_count > 0:
                    helmet_metric.metric("✅ Helmet On", int(helmet_count))
                else:
                    helmet_metric.empty()

                if helmet_count == 0 and no_helmet_count > 0:
                    no_helmet_metric.metric("🚨 No Helmet", int(no_helmet_count))
                    alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")

                    if not alert_sent_state["sent"]:
                        send_telegram_alert(f"⚠️ ALERT: {no_helmet_count} rider(s) without helmet detected!")
                        alert_sent_state["sent"] = True
                else:
                    no_helmet_metric.empty()
                    alert_placeholder.empty()
                    alert_sent_state["sent"] = False

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")

                time.sleep(0.03)

            cap.release()
    else:
        st.info("⬆️ Please upload a video to begin helmet detection.")

else:  # Webcam Mode

    alert_state = {"no_helmet": False, "helmet_count": 0, "no_helmet_count": 0}
    alert_sent_state["sent"] = False

    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            df = results.pandas().xyxy[0]

            helmet_count = (df['name'] == 'helmet_on').sum()
            no_helmet_count = (df['name'] == 'no_helmet').sum()

            alert_state["helmet_count"] = helmet_count
            alert_state["no_helmet_count"] = no_helmet_count
            alert_state["no_helmet"] = (helmet_count == 0 and no_helmet_count > 0)

            img = draw_boxes(img, df)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="helmet-detection",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    def update_ui():
        while True:
            if webrtc_ctx.state.playing:
                helmet_count = alert_state["helmet_count"]
                no_helmet_count = alert_state["no_helmet_count"]
                no_helmet = alert_state["no_helmet"]

                if helmet_count > 0:
                    helmet_metric.metric("✅ Helmet On", int(helmet_count))
                else:
                    helmet_metric.empty()

                if no_helmet:
                    no_helmet_metric.metric("🚨 No Helmet", int(no_helmet_count))
                    alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")

                    if not alert_sent_state["sent"]:
                        send_telegram_alert(f"⚠️ ALERT: {no_helmet_count} rider(s) without helmet detected!")
                        alert_sent_state["sent"] = True
                else:
                    no_helmet_metric.empty()
                    alert_placeholder.empty()
                    alert_sent_state["sent"] = False
            else:
                helmet_metric.empty()
                no_helmet_metric.empty()
                alert_placeholder.info("📷 Webcam inactive.")
                alert_sent_state["sent"] = False
            time.sleep(0.5)

    ui_thread = threading.Thread(target=update_ui, daemon=True)
    ui_thread.start()
