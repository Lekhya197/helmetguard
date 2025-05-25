import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO

st.set_page_config(page_title="HelmetGuard AI", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🎥 HelmetGuard AI - Real-Time Helmet Detection")

# Add a confidence threshold slider in the sidebar
CONF_THRESH = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

def draw_boxes(frame, df):
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = f"{row['name']} {row['confidence']:.2f}"
        color = (0, 255, 0) if row['name'] == 'helmet_on' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

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
        helmet_metric = st.sidebar.empty()
        no_helmet_metric = st.sidebar.empty()
        alert_placeholder = st.sidebar.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.info("🎬 Video processing complete.")
                break

            results = model(frame)
            df = results.pandas().xyxy[0]

            # Apply confidence threshold filter here
            df = df[df['confidence'] >= CONF_THRESH]

            helmet_count = (df['name'] == 'helmet_on').sum()
            no_helmet_count = (df['name'] == 'no_helmet').sum()

            frame = draw_boxes(frame, df)

            helmet_metric.metric("✅ Helmet On", int(helmet_count))
            no_helmet_metric.metric("🚨 No Helmet", int(no_helmet_count))

            if no_helmet_count > 0:
                alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
            else:
                alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            time.sleep(0.03)

        cap.release()
else:
    st.info("⬆️ Please upload a video to begin helmet detection.")
