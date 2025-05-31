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
from collections import defaultdict

# Telegram Bot Info
BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
CHAT_ID = '6674142283'

# Streamlit Config
st.set_page_config(page_title="HelmetGuard AI - YOLOv5", layout="wide")
st.title("🪖 HelmetGuard AI - Helmet Detection with YOLOv5")

# Initialize pygame mixer
pygame.mixer.init()

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

# Sidebar configuration
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
alert_threshold = st.sidebar.slider("Alert Threshold (seconds)", 1, 10, 3)

# Alert State
class AlertManager:
    def __init__(self):
        self.alerted_ids = set()  # Track IDs we've already alerted for
        self.last_alert_time = 0
        self.cooldown = 30  # seconds between alerts for the same ID
        self.helmet_detected_in_session = False
    
    def should_alert(self, current_detections):
        """Determine if we should send an alert for the current detections"""
        current_time = time.time()
        no_helmet_detected = any(d['label'] == 'no_helmet' for d in current_detections)
        
        if not no_helmet_detected:
            return False
            
        # For each detection, check if we need to alert
        needs_alert = False
        for det in current_detections:
            if det['label'] == 'no_helmet' and det['id'] not in self.alerted_ids:
                needs_alert = True
                self.alerted_ids.add(det['id'])
                self.last_alert_time = current_time
                
        return needs_alert
    
    def update_helmet_status(self, has_helmet):
        """Track if we've seen any helmets in this session"""
        if has_helmet:
            self.helmet_detected_in_session = True

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
    try:
        pygame.mixer.music.load("alert.mp3")
        pygame.mixer.music.play()
    except Exception as e:
        st.warning(f"Audio error: {e}")

def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['confidence']
        color = (0, 255, 0) if label == 'helmet_on' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def process_detections(results):
    """Convert model results to our detection format with tracking IDs"""
    detections = []
    for *box, conf, cls in results.xyxy[0]:
        if conf < CONFIDENCE_THRESHOLD:
            continue
        label = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, box)
        
        # Create a simple ID based on position and size (for basic tracking)
        obj_id = hash((x1, y1, x2-x1, y2-y1))
        
        detections.append({
            'box': (x1, y1, x2, y2),
            'label': label,
            'confidence': float(conf),
            'id': obj_id
        })
    
    return detections

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
                
                # Process frame
                results = model(frame)
                detections = process_detections(results)
                
                # Count helmets
                helmet_count = sum(1 for d in detections if d['label'] == 'helmet_on')
                no_helmet_count = sum(1 for d in detections if d['label'] == 'no_helmet')
                
                # Update helmet status
                alert_manager.update_helmet_status(helmet_count > 0)
                
                # Draw boxes
                frame = draw_boxes(frame, detections)
                
                # Update metrics
                helmet_metric.metric("✅ Helmet On", helmet_count)
                no_helmet_metric.metric("🚨 No Helmet", no_helmet_count)
                
                # Check for alerts
                if alert_manager.should_alert(detections):
                    alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                    audio_placeholder.audio("alert.mp3", format="audio/mp3", start_time=0)
                    send_telegram_message_async("🚨 Helmet violation detected in uploaded video!")
                else:
                    # Only show "all clear" if we've seen helmets before
                    if alert_manager.helmet_detected_in_session:
                        alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                    else:
                        alert_placeholder.info("🔍 Detecting...")
                    audio_placeholder.empty()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
                time.sleep(0.03)
            cap.release()
    else:
        st.info("⬆️ Please upload a video to begin helmet detection.")

# ------------------ Webcam ------------------
else:
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            super().__init__()
            self.last_alert_time = 0
            self.alert_cooldown = 5  # seconds
            self.consecutive_no_helmet_frames = 0
        
        def recv(self, frame):
            img_bgr = frame.to_ndarray(format="bgr24")
            img_bgr = cv2.resize(img_bgr, (1280, 720))
            
            # Process frame
            results = model(img_bgr)
            detections = process_detections(results)
            
            # Count helmets
            helmet_count = sum(1 for d in detections if d['label'] == 'helmet_on')
            no_helmet_count = sum(1 for d in detections if d['label'] == 'no_helmet')
            
            # Update helmet status
            alert_manager.update_helmet_status(helmet_count > 0)
            
            # Track consecutive no-helmet frames
            if no_helmet_count > 0:
                self.consecutive_no_helmet_frames += 1
            else:
                self.consecutive_no_helmet_frames = 0
            
            # Check for alerts
            current_time = time.time()
            if (self.consecutive_no_helmet_frames >= alert_threshold and 
                alert_manager.should_alert(detections) and
                current_time - self.last_alert_time > self.alert_cooldown):
                
                send_telegram_message_async("🚨 Helmet violation detected in webcam feed!")
                play_alert_sound()
                self.last_alert_time = current_time
            
            # Draw boxes and convert to RGB
            img_bgr = draw_boxes(img_bgr, detections)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            return av.VideoFrame.from_ndarray(img_rgb, format="rgb24")

    alert_placeholder = st.empty()
    audio_placeholder = st.empty()
    helmet_metric = st.sidebar.empty()
    no_helmet_metric = st.sidebar.empty()

    ctx = webrtc_streamer(
        key="helmet-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    def update_ui():
        while True:
            if ctx.video_processor:
                processor = ctx.video_processor
                
                # Update metrics (placeholder values since we can't access processor directly)
                helmet_metric.metric("✅ Helmet On", "N/A")
                no_helmet_metric.metric("🚨 No Helmet", "N/A")
                
                if processor.consecutive_no_helmet_frames >= alert_threshold:
                    alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                    audio_placeholder.audio("alert.mp3", format="audio/mp3", start_time=0)
                else:
                    if alert_manager.helmet_detected_in_session:
                        alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                    else:
                        alert_placeholder.info("🔍 Detecting...")
                    audio_placeholder.empty()
            else:
                alert_placeholder.info("📷 Webcam inactive.")
                audio_placeholder.empty()
                helmet_metric.metric("✅ Helmet On", 0)
                no_helmet_metric.metric("🚨 No Helmet", 0)
            
            time.sleep(0.5)

    threading.Thread(target=update_ui, daemon=True).start()
