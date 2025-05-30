import streamlit as st
import torch
import numpy as np
import cv2
import requests
import threading
import pygame
from PIL import Image

# Initialize Telegram bot config
BOT_TOKEN = '7133866876:AAFXl8AAKLCxQxgzdpOeBItLBh3ndAkt46Y'
CHAT_ID = '6674142283'

# Async Telegram alert
def send_telegram_message_async(message):
    def send():
        send_telegram_message(message)
    threading.Thread(target=send).start()

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {'chat_id': CHAT_ID, 'text': message}
    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            st.warning(f"Failed to send Telegram message: {response.text}")
    except Exception as e:
        st.warning(f"Error sending Telegram message: {e}")

# Alert function
def alert_no_helmet():
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()
    send_telegram_message_async("🚨 Helmet violation detected!")

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

# Streamlit UI
st.title("🪖 Real-Time Helmet Detection")
run = st.checkbox('Start Webcam')

# Parameters
threshold = 3
no_helmet_conf_threshold = 0.3
default_conf_threshold = 0.3
no_helmet_count = 0

frame_window = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break

        frame = cv2.resize(frame, (1020, 600))
        results = model(frame)
        df = results.pandas().xyxy[0]

        filtered = df[
            ((df['name'] == 'no_helmet') & (df['confidence'] >= no_helmet_conf_threshold)) |
            ((df['name'] != 'no_helmet') & (df['confidence'] >= default_conf_threshold))
        ]

        labels = list(filtered['name'])

        if 'no_helmet' in labels:
            no_helmet_count += 1
        else:
            no_helmet_count = 0

        if no_helmet_count >= threshold:
            alert_no_helmet()
            no_helmet_count = 0

        for _, row in filtered.iterrows():
            xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = f"{row['name']} {row['confidence']:.2f}"

            if row['name'] == 'helmet_on':
                color = (0, 255, 0)
            elif row['name'] == 'no_helmet':
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

    cap.release()
else:
    st.info("👆 Turn on the checkbox to start the helmet detection webcam.")
