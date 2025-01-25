import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def process_and_annotate(image, threshold):
    """Process the image and annotate it with pose landmarks."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    annotated_image = image.copy()

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility > threshold:  # Adjust based on threshold
                h, w, _ = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(annotated_image, (cx, cy), 5, (0, 255, 0), -1)
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),  # Change color here
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)   # Change color here
        )

    return annotated_image

def main():
    st.set_page_config(page_title="Human Pose Estimator", page_icon="🕺", layout="wide")

    # Header Section
    st.markdown("""
    <style>
    .header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f4f4f4;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 1rem;
        color: #aaa;
    }
    </style>
    <div class="header">🕺 Human Pose Estimator</div>
    <div class="subheader">Transform your images into dynamic pose visualizations with cutting-edge AI</div>
    """, unsafe_allow_html=True)

    # Sidebar Section
    st.sidebar.header("Upload Your Media")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.1)

    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated_image = process_and_annotate(image_bgr, threshold)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        st.write("### Results")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(annotated_image_rgb, caption="Pose Annotated Image", use_column_width=True)

        st.success("Pose analysis for image completed!")

    else:
        st.info("Upload an image from the sidebar to begin pose estimation.")

    # Additional Sections
    st.markdown("""
    ---
    ### About Pose Estimation
    Human Pose Estimator is a state-of-the-art AI-powered pose analyzer that brings your images to life by identifying and visualizing key human body points such as shoulders, elbows, and knees.
    It has applications in:
    - Fitness Tracking : Enhance your workout sessions with precise form correction.
    - Sports Analytics : Improve athletic performance through detailed motion analysis.
    - Healthcare & Rehabilitation : Assist patients with their recovery exercises.
    - Augmented Reality : Create immersive AR experiences by integrating real-time pose tracking.

    ---
    ### How It Works
    1. Upload an image in the sidebar.
    2. Select the desired threshold for pose keypoint detection.
    3. Our model processes the media using [Mediapipe](https://mediapipe.dev/).
    4. The annotated image is displayed alongside the original.

    <div class="footer">
        Crafted with 💙 by Abhishek Jha to make AI accessible and insightful.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

pose.close()
