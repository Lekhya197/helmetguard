import streamlit as st
import torch
import cv2
import numpy as np

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)

model = load_model()

st.title("🎥 HelmetGuard AI - Real-Time Helmet Detection")

def draw_boxes(frame, df):
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = f"{row['name']} {row['confidence']:.2f}"
        color = (0, 255, 0) if row['name'] == 'helmet_on' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

frame_placeholder = st.empty()

cap = cv2.VideoCapture("/Users/sreelekhyauggina/Desktop/project/he2.mp4")

if not cap.isOpened():
    st.error("Cannot open video file: /Users/sreelekhyauggina/Desktop/project/he2.mp4")
else:
    st.warning("Streaming from video file")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame. Stream ended or cannot capture.")
            break

        results = model(frame)
        df = results.pandas().xyxy[0]

        helmet_count = (df['name'] == 'helmet_on').sum()
        no_helmet_count = (df['name'] == 'no_helmet').sum()

        frame = draw_boxes(frame, df)

        st.sidebar.metric("✅ Helmet On", helmet_count)
        st.sidebar.metric("🚨 No Helmet", no_helmet_count)

        if no_helmet_count > 0:
            st.sidebar.error("Alert: No helmet detected!")
        else:
            st.sidebar.success("All good: Helmets detected.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

cap.release()
