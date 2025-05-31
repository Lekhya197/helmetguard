import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import torch
import av
import threading
import numpy as np
import time

st.set_page_config(page_title="HelmetGuard AI - YOLOv5", layout="wide")

@st.cache_resource(show_spinner=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

alert_audio_file = open("alert.mp3", "rb").read()

st.title("🎥 HelmetGuard AI - YOLOv5 Helmet Detection")

mode = st.sidebar.radio("Select Mode", ["Upload Video", "Webcam"])

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

alert_placeholder = st.empty()
audio_placeholder = st.empty()

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
            frame_placeholder = st.empty()
            helmet_metric = st.sidebar.empty()
            no_helmet_metric = st.sidebar.empty()
            alert_placeholder = st.sidebar.empty()
            audio_placeholder = st.empty()

            alert_triggered = False  # Track alert state

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

                # Always show helmet count (even if zero)
                helmet_metric.metric("✅ Helmet On", helmet_count)

                # Show no helmet count only if > 0, else clear
                if no_helmet_count > 0:
                    no_helmet_metric.metric("🚨 No Helmet", no_helmet_count)
                    if not alert_triggered:
                        alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                        audio_placeholder.audio(alert_audio_file, format="audio/mp3", start_time=0)
                        alert_triggered = True
                else:
                    no_helmet_metric.empty()
                    if alert_triggered:
                        alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                        audio_placeholder.empty()
                        alert_triggered = False

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")

                time.sleep(0.03)
            cap.release()

    else:
        st.info("⬆️ Please upload a video to begin helmet detection.")

else:  # Webcam mode

    alert_state = {"no_helmet": False}

    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            labels = [model.names[int(cls)] for cls in results.xyxy[0][:, 5]]
            alert_state["no_helmet"] = labels.count('no_helmet') > 0
            alert_state["helmet_on"] = labels.count('helmet_on')
            img = draw_boxes(img, results)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(key="helmet-detection", video_processor_factory=VideoProcessor,
                                 media_stream_constraints={"video": True, "audio": False},
                                 async_processing=True)

    helmet_metric = st.sidebar.empty()
    no_helmet_metric = st.sidebar.empty()

    def update_ui():
        alert_triggered = False
        while True:
            if webrtc_ctx.state.playing:
                helmet_metric.metric("✅ Helmet On", alert_state.get("helmet_on", 0))

                if alert_state["no_helmet"]:
                    no_helmet_count = 1  # just show alert and count 1, or customize as needed
                    no_helmet_metric.metric("🚨 No Helmet", no_helmet_count)
                    if not alert_triggered:
                        alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                        audio_placeholder.audio(alert_audio_file, format="audio/mp3", start_time=0)
                        alert_triggered = True
                else:
                    no_helmet_metric.empty()
                    if alert_triggered:
                        alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                        audio_placeholder.empty()
                        alert_triggered = False
            else:
                helmet_metric.empty()
                no_helmet_metric.empty()
                alert_placeholder.info("📷 Webcam inactive.")
                audio_placeholder.empty()
            time.sleep(0.5)

    thread = threading.Thread(target=update_ui, daemon=True)
    thread.start()
