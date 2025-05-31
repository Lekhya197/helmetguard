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
import math

BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
CHAT_ID = '6674142283'

st.set_page_config(page_title="HelmetGuard AI - YOLOv5", layout="wide")
st.title("🪖 HelmetGuard AI - Helmet Detection with YOLOv5")

alert_audio_file = open("alert.mp3", "rb").read()

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
alert_threshold = 3  # Frames with detection to trigger alert

class AlertManager:
    def __init__(self):
        self.triggered_ids = set()
        self.helmet_previously_detected = False

    def _center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _is_new_detection(self, center, known_centers, distance_threshold=50):
        for c in known_centers:
            dist = math.hypot(center[0] - c[0], center[1] - c[1])
            if dist < distance_threshold:
                return False
        return True

    def update_and_check_alert(self, detections):
        no_helmet_boxes = [box for box, label in detections if label == 'no_helmet']
        helmet_present = any(label == 'helmet_on' for _, label in detections)

        if helmet_present:
            self.helmet_previously_detected = True

        alert_needed = False

        # Track new no_helmet detections by center location
        new_ids = set()
        known_centers = [self._center(box) for box in self.triggered_ids]

        for box in no_helmet_boxes:
            center = self._center(box)
            if self._is_new_detection(center, known_centers):
                new_ids.add(tuple(box))
                known_centers.append(center)

        if new_ids and self.helmet_previously_detected:
            alert_needed = True
            self.triggered_ids.update(new_ids)
            self.helmet_previously_detected = False

        # Reset if no no_helmet detected at all
        if not no_helmet_boxes:
            self.triggered_ids.clear()
            alert_needed = False

        return alert_needed

alert_manager = AlertManager()

def send_telegram_message_async(message):
    def send():
        try:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          data={'chat_id': CHAT_ID, 'text': message})
        except Exception as e:
            st.warning(f"Telegram error: {e}")
    threading.Thread(target=send).start()

def play_alert_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()

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
                detections_raw = results.xyxy[0]
                filtered_detections = [(list(map(int, box[:4])), model.names[int(cls)]) 
                                       for *box, conf, cls in detections_raw if conf >= CONFIDENCE_THRESHOLD]
                labels = [label for _, label in filtered_detections]

                helmet_count = labels.count('helmet_on')
                no_helmet_count = labels.count('no_helmet')

                frame = draw_boxes(frame, results)

                helmet_metric.metric("✅ Helmet On", helmet_count)
                no_helmet_metric.metric("🚨 No Helmet", no_helmet_count)

                if alert_manager.update_and_check_alert(filtered_detections):
                    alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                    audio_placeholder.audio(alert_audio_file, format="audio/mp3", start_time=0)
                    send_telegram_message_async("🚨 Helmet violation detected in uploaded video!")
                else:
                    # Show all clear only if no no_helmet detected
                    if no_helmet_count == 0:
                        alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                        audio_placeholder.empty()
                    else:
                        alert_placeholder.info("⚠️ Detecting, waiting for stable alert...")

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
                time.sleep(0.03)
            cap.release()
    else:
        st.info("⬆️ Please upload a video to begin helmet detection.")

else:
    alert_state = {
        "no_helmet_centers": set(),
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
                ((df['name'] == 'no_helmet') & (df['confidence'] >= CONFIDENCE_THRESHOLD)) |
                ((df['name'] != 'no_helmet') & (df['confidence'] >= CONFIDENCE_THRESHOLD))
            ]

            detections = []
            for _, row in filtered.iterrows():
                box = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                label = row['name']
                detections.append((box, label))

            no_helmet_boxes = [box for box, label in detections if label == 'no_helmet']
            helmet_present = any(label == 'helmet_on' for _, label in detections)

            def center(box):
                x1, y1, x2, y2 = box
                return ((x1 + x2) // 2, (y1 + y2) // 2)

            def is_new(center, known_centers, dist_threshold=50):
                for c in known_centers:
                    dist = math.hypot(center[0] - c[0], center[1] - c[1])
                    if dist < dist_threshold:
                        return False
                return True

            new_centers = set()
            for box in no_helmet_boxes:
                c = center(box)
                if is_new(c, alert_state["no_helmet_centers"]):
                    new_centers.add(c)

            if new_centers and helmet_present:
                alert_state["no_helmet_centers"].update(new_centers)
                alert_state["no_helmet_count"] = 0  # reset frame count after new detection
                send_telegram_message_async("🚨 Helmet violation detected in webcam feed!")
                play_alert_sound()

            if no_helmet_boxes:
                alert_state["no_helmet_count"] += 1
            else:
                alert_state["no_helmet_count"] = 0
                alert_state["no_helmet_centers"].clear()  # reset when no no_helmet detected

            for box, label in detections:
                xmin, ymin, xmax, ymax = box
                color = (0, 255, 0) if label == 'helmet_on' else (0, 0, 255)
                label_text = f"{label} {round(df.loc[(df['xmin'] == xmin) & (df['ymin'] == ymin), 'confidence'].values[0], 2):.2f}"
                cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img_bgr, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
                if alert_state["no_helmet_centers"]:
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
