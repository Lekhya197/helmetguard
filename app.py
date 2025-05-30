import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import torch
import cv2
import numpy as np
import pandas as pd
import pygame
import threading
import requests

# Configuration
BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
CHAT_ID = '6674142283'
THRESHOLD = 3
NO_HELMET_CONF_THRESHOLD = 0.1
DEFAULT_CONF_THRESHOLD = 0.1

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    model.conf = 0.25  # Confidence threshold
    model.iou = 0.45   # IoU threshold
    return model

# Alert system
def send_telegram_message_async(message):
    def send():
        send_telegram_message(message)
    threading.Thread(target=send).start()

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, data={'chat_id': CHAT_ID, 'text': message})
    except Exception as e:
        st.warning(f"Telegram error: {e}")

def alert_no_helmet():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alert.mp3")
        pygame.mixer.music.play()
        send_telegram_message_async("🚨 Helmet violation detected!")
    except Exception as e:
        st.warning(f"Alert error: {e}")

# Streamlit UI
st.title("🪖 Helmet Detection with YOLOv5 + Webcam")
st.write("Ensure your webcam has permission to access this page.")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.no_helmet_count = 0
        self.model = load_model()
        self.model.eval()
        
    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (640, 640))
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Inference
            results = self.model(img_rgb)
            df = results.pandas().xyxy[0]
            
            # Filter results
            filtered = df[
                ((df['name'] == 'no_helmet') & (df['confidence'] >= NO_HELMET_CONF_THRESHOLD)) |
                ((df['name'] != 'no_helmet') & (df['confidence'] >= DEFAULT_CONF_THRESHOLD))
            ]
            
            # Update helmet violation count
            if 'no_helmet' in filtered['name'].values:
                self.no_helmet_count += 1
            else:
                self.no_helmet_count = max(0, self.no_helmet_count - 1)
            
            # Trigger alert if threshold reached
            if self.no_helmet_count >= THRESHOLD:
                alert_no_helmet()
                self.no_helmet_count = 0
            
            # Draw detections
            for _, row in filtered.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                label = f"{row['name']} {row['confidence']:.2f}"
                color = (0, 255, 0) if row['name'] == 'helmet_on' else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            st.error(f"Processing error: {e}")
            return frame

# WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="helmet-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 640}
        },
        "audio": False
    },
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

if not webrtc_ctx.state.playing:
    st.info("Waiting for webcam to start...")
