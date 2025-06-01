import numpy as np
import streamlit as st
import cv2
from collections import deque
import os
import subprocess
from keras.models import load_model

# Constants
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20

CLASSES_LIST = [
    "BaseballPitch", "Basketball", 'BenchPress', "Biking", "JavelinThrow", "CleanAndJerk",
    "Diving", "Billiards", "HighJump", "HorseRace", "MilitaryParade", "PlayingGuitar",
    "ThrowDiscus", "WalkingWithDog", "SkateBoarding", "null", "JumpingJack", "JumpRope",
    "Kayaking", "HulaHoop", "JugglingBalls", "GolfSwing", "Fencing", "Drumming",
    "HorseRiding", "BreastStroke"
]

@st.cache_resource
def load_LRCN_model():
    return load_model("model1.h5")

def predict_single_action(video_path, model):
    video_reader = cv2.VideoCapture(video_path)
    frames_list = []

    frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_window = max(int(frame_count / SEQUENCE_LENGTH), 1)

    for i in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, i * skip_window)
        success, frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    predicted_probs = model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_probs)
    predicted_class_name = CLASSES_LIST[predicted_label]

    result = f"Action Predicted: {predicted_class_name} | Confidence: {predicted_probs[predicted_label]:.2f}"
    video_reader.release()
    return result

def annotate_video(video_path, output_path, model):
    video_reader = cv2.VideoCapture(video_path)
    width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class = ''

    while video_reader.isOpened():
        ret, frame = video_reader.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_queue.append(normalized_frame)

        if len(frames_queue) == SEQUENCE_LENGTH:
            predicted_probs = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_probs)
            predicted_class = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        video_writer.write(frame)

    video_reader.release()
    video_writer.release()

def main():
    st.set_page_config(page_title="🎥 Action Recognition", layout="centered", page_icon="🎬")
    st.title("🎥 Action Recognition Web App")

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mpeg"])

    if uploaded_file is not None:
        os.makedirs("temp", exist_ok=True)
        os.makedirs("video", exist_ok=True)

        temp_video_path = os.path.join("temp", uploaded_file.name)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Video uploaded successfully!")

        # Show uploaded video preview
        st.video(temp_video_path)

        if st.button("Classify the Video"):
            st.info("⏳ Processing...")
            model = load_LRCN_model()
            result = predict_single_action(temp_video_path, model)
            st.success(result)

            output_path = os.path.join("video", uploaded_file.name.split('.')[0] + "_output.mp4")
            annotate_video(temp_video_path, output_path, model)

            # Re-encode video using ffmpeg (optional)
            subprocess.call([
                'ffmpeg', '-y', '-i', output_path,
                '-vcodec', 'libx264', '-f', 'mp4', 'output_final.mp4'
            ], shell=True)

            st.video("output_final.mp4")

if __name__ == "__main__":
    main()
