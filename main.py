import streamlit as st
import cv2
import time
import base64
import numpy as np
import pyttsx3
import threading
from groq import Groq
from PIL import Image
import tempfile
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq API client
client = Groq(api_key=GROQ_API_KEY)


# Function to encode image to base64
def encode_image(image_path):
    """Encodes an image to base64 format."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to send image to Groq API and get detected sign
def detect_sign(image_path):
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "You are an expert sign language detector. Identify the sign in the image and provide a short, clear response with the detected word or phrase only."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0.7,
        max_completion_tokens=512
    )

    return response.choices[0].message.content.strip()


# Function for Text-to-Speech output (Reinitialize engine each time)
def speak_text(text):
    def run_tts():
        engine = pyttsx3.init()  # Reinitialize TTS engine every time
        engine.setProperty('rate', 150)  # Adjust speed
        engine.say(text)
        engine.runAndWait()

    tts_thread = threading.Thread(target=run_tts)
    tts_thread.start()


# Streamlit UI
st.title("🤟 Sign Language Detector")
st.write("Detect and understand sign language gestures in real-time!")

# Display GIF if it exists
gif_path = "hand_sign.gif"
if os.path.exists(gif_path):
    st.image(gif_path, caption="Sign Language Detection in Progress", use_container_width=True)

# Start video capture button
start_button = st.button("📷 Start Video Capture")

if start_button:
    cap = cv2.VideoCapture(0)  # Open webcam
    frames = []
    start_time = time.time()
    duration = 5  # Capture for 5 seconds

    st.write("🎥 Recording for 5 seconds... Perform two different gestures.")

    # Progress bar
    progress_bar = st.progress(0)
    frame_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video frame.")
            break

        # Convert frame to RGB (for Streamlit display)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show live video feed
        st.image(frame_rgb, caption="Live Video Feed", use_container_width=True)

        # Store frames for analysis
        frames.append(frame)
        frame_count += 1

        # Update progress bar
        progress_bar.progress(int((time.time() - start_time) / duration * 100))

    cap.release()
    st.success("✅ Video capture complete!")

    if len(frames) >= 2:  # Ensure we have enough frames for two gestures
        detected_signs = []

        for i in range(2):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                # Select two frames (one from the beginning, one from the end)
                selected_frame = frames[i * len(frames) // 2]

                # Save frame as image
                Image.fromarray(cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)).save(temp_file.name)
                temp_file_path = temp_file.name

                # Detect sign from Groq API
                with st.spinner(f"🔍 Detecting Sign {i + 1}..."):
                    detected_sign = detect_sign(temp_file_path)
                    detected_signs.append(detected_sign)

        # Display detected signs
        st.subheader("🔎 Detected Signs:")
        for i, sign in enumerate(detected_signs):
            st.write(f"**Sign {i + 1}:** {sign}")

        # Speak detected signs aloud
        if detected_signs:
            speak_text(" and ".join(detected_signs))
            st.success("🗣️ Text-to-Speech Output Complete!")
