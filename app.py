import streamlit as st
import cv2
import torch
import numpy as np
import time

st.set_page_config(page_title="HelmetGuard AI - YOLOv5", layout="wide")

@st.cache_resource(show_spinner=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

st.title("🎥 HelmetGuard AI - YOLOv5 Helmet Detection")

CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

alert_audio_file = open("alert.mp3", "rb").read()

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

mode = st.sidebar.radio("Select Input Mode", ["Upload Video", "Webcam"])

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
                else:
                    alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                    audio_placeholder.empty()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")

                time.sleep(0.03)

            cap.release()
    else:
        st.info("⬆️ Please upload a video to begin helmet detection.")

elif mode == "Webcam":
    st.info("📷 Webcam mode activated. Please allow access to your webcam.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not access the webcam. Please ensure your webcam is connected.")
    else:
        frame_placeholder = st.empty()
        helmet_metric = st.sidebar.empty()
        no_helmet_metric = st.sidebar.empty()
        alert_placeholder = st.sidebar.empty()
        audio_placeholder = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture frame from webcam.")
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
            else:
                alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                audio_placeholder.empty()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            time.sleep(0.03)

        cap.release()
