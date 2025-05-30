import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import torch
import av
import numpy as np
import time

st.set_page_config(page_title="HelmetGuard AI - YOLOv5", layout="wide")

@st.cache_resource(show_spinner=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model

model = load_model()
CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
alert_audio_file = open("alert.mp3", "rb").read()

st.title("🛡️ HelmetGuard AI - Helmet Detection")
mode = st.sidebar.radio("Choose Mode", ["Upload Video", "Webcam"])

alert_placeholder = st.empty()
audio_placeholder = st.empty()

def draw_boxes(frame, results):
    for *box, conf, cls in results.xyxy[0]:
        if conf < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        color = (0, 255, 0) if label == 'helmet_on' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ==================== Upload Mode ====================
if mode == "Upload Video":
    video_file = st.file_uploader("📂 Upload a video", type=["mp4", "mov", "avi"])
    if video_file:
        with open("temp.mp4", "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture("temp.mp4")
        placeholder = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            img_resized = cv2.resize(frame, (640, 640))
            results = model(img_resized)
            labels = [model.names[int(cls)] for cls in results.xyxy[0][:, 5]]
            no_helmet = labels.count('no_helmet')

            frame = draw_boxes(frame, results)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            placeholder.image(frame, channels="RGB")
            if no_helmet > 0:
                alert_placeholder.error("🚨 Riders without helmets detected!")
                audio_placeholder.audio(alert_audio_file, format="audio/mp3", start_time=0)
            else:
                alert_placeholder.success("✅ All riders wearing helmets.")
                audio_placeholder.empty()
            time.sleep(0.03)
        cap.release()
    else:
        st.info("Upload a video to start detection.")

# ==================== Webcam Mode ====================
else:
    alert_state = {"no_helmet": False}

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.last_ts = time.time()

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_resized = cv2.resize(img, (640, 640))  # Very important!
            results = model(img_resized)

            labels = [model.names[int(cls)] for cls in results.xyxy[0][:, 5]]
            alert_state["no_helmet"] = labels.count('no_helmet') > 0

            img = draw_boxes(img, results)
            return img

    webrtc_ctx = webrtc_streamer(
        key="helmet-detect",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    def update_alerts():
        while True:
            if webrtc_ctx.state.playing:
                if alert_state["no_helmet"]:
                    alert_placeholder.error("🚨 No helmet detected!")
                    audio_placeholder.audio(alert_audio_file, format="audio/mp3", start_time=0)
                else:
                    alert_placeholder.success("✅ All riders wearing helmets.")
                    audio_placeholder.empty()
            time.sleep(0.5)

    import threading
    threading.Thread(target=update_alerts, daemon=True).start()
