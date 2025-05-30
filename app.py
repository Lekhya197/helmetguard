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

BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
CHAT_ID = '6674142283'

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

# Alert logic
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
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()
    send_telegram_message_async("🚨 Helmet violation detected!")

# Streamlit UI
st.title("🪖 Helmet Detection with YOLOv5 + Webcam")

threshold = 3
no_helmet_conf_threshold = 0.1  # Lowered threshold
default_conf_threshold = 0.1    # Lowered threshold

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.no_helmet_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (1020, 600))

        results = model(img)
        df = results.pandas().xyxy[0]

        print("Frame shape:", img.shape)
        print("Detected classes:", df['name'].unique())
        print(df.head())

        filtered = df[
            ((df['name'] == 'no_helmet') & (df['confidence'] >= no_helmet_conf_threshold)) |
            ((df['name'] != 'no_helmet') & (df['confidence'] >= default_conf_threshold))
        ]

        labels = list(filtered['name'])

        if 'no_helmet' in labels:
            self.no_helmet_count += 1
        else:
            self.no_helmet_count = 0

        if self.no_helmet_count >= threshold:
            alert_no_helmet()
            self.no_helmet_count = 0

        for _, row in filtered.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"
            color = (0, 255, 0) if row['name'] == 'helmet_on' else (0, 0, 255)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(img, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="helmet-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True)
