import streamlit as st
import torch
import cv2
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="HelmetGuard AI YOLOv5", layout="wide")

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

st.title("🎥 HelmetGuard AI - YOLOv5 Helmet Detection")

def draw_boxes(frame, results_df):
    for _, row in results_df.iterrows():
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

            helmet_count = (df['name'] == 'helmet_on').sum()
            no_helmet_count = (df['name'] == 'no_helmet').sum()

            frame = draw_boxes(frame, df)

            # Show helmet count only if > 0
            if helmet_count > 0:
                helmet_metric.metric("✅ Helmet On", int(helmet_count))
            else:
                helmet_metric.empty()

            # Show no helmet count & alert only if helmets == 0 and no helmets > 0
            if helmet_count == 0 and no_helmet_count > 0:
                no_helmet_metric.metric("🚨 No Helmet", int(no_helmet_count))
                alert_placeholder.error("⚠️ Alert: Riders without helmets detected!")
            else:
                no_helmet_metric.empty()
                alert_placeholder.empty()

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
