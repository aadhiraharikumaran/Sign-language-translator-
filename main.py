import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client (do this once)
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=API_KEY)

# Function to detect sign language using Gemini API
def detect_sign_language(frame, client):
    try:
        # Convert frame to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
      
        # Send image to Gemini API
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                "Identify the sign language gesture shown in this image. "
                "Return the name of the sign and a confidence score between 0 and 1.",
                pil_image
            ]
        )
      
        # Parse response (assuming Gemini returns "Sign: Hello, Confidence: 0.95")
        response_text = response.text.strip()
        sign = response_text.split("Sign: ")[1].split(",")[0]
        confidence = float(response_text.split("Confidence: ")[1])
      
        return sign, confidence
  
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def main():
    st.title("Sign Language Translator")
  
    # Get Gemini client
    client = get_gemini_client()
  
    # Initialize session state for camera
    if 'run' not in st.session_state:
        st.session_state.run = False
  
    # Start/Stop button
    if st.button("Start Camera" if not st.session_state.run else "Stop Camera"):
        st.session_state.run = not st.session_state.run
  
    # Create placeholders for video and prediction
    video_placeholder = st.empty()
    prediction_placeholder = st.empty()
  
    # Initialize webcam
    cap = cv2.VideoCapture(0)
  
    # Rate limiting for API calls
    last_processed_time = 0
    process_interval = 1  # Process every 1 second to avoid API rate limits
  
    while st.session_state.run:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
      
        # Display the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")
      
        # Process frame periodically
        current_time = time.time()
        if current_time - last_processed_time >= process_interval:
            sign, confidence = detect_sign_language(frame, client)
            if "Error" in sign:
                prediction_placeholder.error(sign)
            else:
                prediction_placeholder.write(
                    f"Predicted Sign: {sign} (Confidence: {confidence:.2f})"
                )
            last_processed_time = current_time
      
        # Small delay to prevent overwhelming the interface
        time.sleep(0.033)  # ~30 fps
  
    # Release the webcam when stopped
    cap.release()

if __name__ == "__main__":
    main()
