import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import torch
import av
import threading

st.set_page_config(page_title="HelmetGuard AI - YOLOv5", layout="wide")

@st.cache_resource(show_spinner=True)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

CONFIDENCE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

alert_audio_file = open("alert.mp3", "rb").read()

st.title("🎥 HelmetGuard AI - YOLOv5 Helmet Detection")

alert_state = {"no_helmet": False}

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

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        labels = [model.names[int(cls)] for cls in results.xyxy[0][:, 5]]
        alert_state["no_helmet"] = labels.count('no_helmet') > 0
        img = draw_boxes(img, results)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

alert_placeholder = st.empty()
audio_placeholder = st.empty()

webrtc_ctx = webrtc_streamer(key="helmet-detection", video_processor_factory=VideoProcessor,
                             media_stream_constraints={"video": True, "audio": False},
                             async_processing=True)

def update_ui():
    import time
    while True:
        if webrtc_ctx.state.playing:
            if alert_state["no_helmet"]:
                alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
                audio_placeholder.audio(alert_audio_file, format="audio/mp3", start_time=0)
            else:
                alert_placeholder.success("🟢 All Clear: All riders wearing helmets.")
                audio_placeholder.empty()
        else:
            alert_placeholder.info("📷 Webcam inactive.")
            audio_placeholder.empty()
        time.sleep(0.5)

thread = threading.Thread(target=update_ui, daemon=True)
thread.start()
