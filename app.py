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

# Telegram Bot Info
BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
CHAT_ID = '6674142283'

# Load model only once
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

# Send Telegram alert (async)
def send_telegram_message_async(message):
    def send():
        try:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                          data={'chat_id': CHAT_ID, 'text': message})
        except Exception as e:
            st.warning(f"Telegram error: {e}")
    threading.Thread(target=send).start()

# Play alert sound and notify
def alert_no_helmet():
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()
    send_telegram_message_async("🚨 Helmet violation detected!")

# Streamlit Title
st.title("🪖 Helmet Detection with YOLOv5 + Webcam")

# Thresholds
alert_threshold = 3
no_helmet_conf_threshold = 0.05
default_conf_threshold = 0.05

# Video Processor Class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.no_helmet_count = 0

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_bgr = cv2.resize(img_bgr, (1280, 720))  # Resize if needed
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        results = model(img_rgb)
        df = results.pandas().xyxy[0]

        filtered = df[
            ((df['name'] == 'no_helmet') & (df['confidence'] >= no_helmet_conf_threshold)) |
            ((df['name'] != 'no_helmet') & (df['confidence'] >= default_conf_threshold))
        ]

        labels = list(filtered['name'])

        if 'no_helmet' in labels:
            self.no_helmet_count += 1
        else:
            self.no_helmet_count = 0

        if self.no_helmet_count >= alert_threshold:
            alert_no_helmet()
            self.no_helmet_count = 0

        for _, row in filtered.iterrows():
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = f"{row['name']} {row['confidence']:.2f}"
            color = (0, 255, 0) if row['name'] == 'helmet_on' else (0, 0, 255)
            cv2.rectangle(img_bgr, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(img_bgr, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# Start Webcam Stream
webrtc_streamer(
    key="helmet-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
