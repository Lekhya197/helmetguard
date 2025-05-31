import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import torch
import cv2
import av
import numpy as np
import time
import threading
import requests

# Telegram Bot Info
BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
CHAT_ID = '6674142283'

# Streamlit Config
st.set_page_config(page_title="HelmetGuard AI - YOLOv5", layout="wide")
st.title("🪖 HelmetGuard AI - Helmet Detection with YOLOv5")

# Load alert audio once
alert_audio_file = open("alert.mp3", "rb").read()

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

# Sidebar configuration
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
alert_threshold = 3  # number of consecutive frames to confirm alert

# Telegram message sending (async)
def send_telegram_message_async(message):
    def send():
        try:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          data={'chat_id': CHAT_ID, 'text': message})
        except Exception as e:
            st.warning(f"Telegram error: {e}")
    threading.Thread(target=send).start()

# Draw bounding boxes on frame
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

# Mode selection
mode = st.sidebar.radio("Select Mode", ["Upload Video", "Webcam"])

# ------------------ Upload Video ------------------
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

                if no_helmet_count > 0:
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

# ------------------ Webcam ------------------
else:
    # Store object states by track_id
    object_states = {}

    # Frame count for alert threshold
    alert_frame_counter = 0

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.alert_frame_counter = 0
        self.object_states = {}

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_bgr = cv2.resize(img_bgr, (1280, 720))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = model(img_rgb)
        df = results.pandas().xyxy[0]

        filtered = df[df['confidence'] >= CONFIDENCE_THRESHOLD]

        detections = []
        labels = []

        for i, row in filtered.iterrows():
            label = row['name']
            conf = row['confidence']
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            track_id = i  # Replace with actual tracker ID if available
            detections.append({'id': track_id, 'label': label, 'bbox': (xmin, ymin, xmax, ymax)})
            labels.append(label)

        violation_detected_this_frame = False

        for detection in detections:
            track_id = int(detection['id'])
            label = detection['label']

            state = self.object_states.get(track_id, {'last_state': None, 'alert_sent': False})

            if label == 'no_helmet' and state['last_state'] == 'helmet_on' and not state['alert_sent']:
                send_telegram_message_async(f"🚨 Helmet violation for object #{track_id}")
                state['alert_sent'] = True
                violation_detected_this_frame = True

            elif label == 'helmet_on':
                state['alert_sent'] = False
                state['last_state'] = 'helmet_on'

            else:
                if state['last_state'] is None:
                    state['last_state'] = label

            self.object_states[track_id] = state

        if violation_detected_this_frame:
            self.alert_frame_counter += 1
        else:
            self.alert_frame_counter = 0

        if self.alert_frame_counter >= alert_threshold:
            st.session_state['alert_active'] = True
        else:
            st.session_state['alert_active'] = False

        for detection in detections:
            xmin, ymin, xmax, ymax = detection['bbox']
            label = detection['label']
            color = (0, 255, 0) if label == 'helmet_on' else (0, 0, 255)
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(img_bgr, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    if 'alert_active' not in st.session_state:
        st.session_state['alert_active'] = False

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
                if st.session_state['alert_active']:
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
