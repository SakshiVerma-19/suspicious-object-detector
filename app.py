import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_option_menu import option_menu

# --- PAGE CONFIGURATION ---
# Sets the page title, icon, and layout for a more professional look.
st.set_page_config(
    page_title="Sharp Object Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL LOADING ---
# Caches the model to prevent reloading on every interaction, improving performance.
@st.cache_resource
def load_model():
    """Loads the YOLOv8 model from the specified path."""
    model = YOLO('runs/detect/train3/weights/best.pt')
    return model

# Load the model and display a spinner while it's loading.
with st.spinner('Loading the detection model...'):
    model = load_model()

# --- SIDEBAR NAVIGATION ---
# Creates a clean, icon-driven navigation menu in the sidebar.
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Live Detection", "About"],
        icons=["house-door-fill", "camera-video-fill", "info-circle-fill"],
        menu_icon="cast",
        default_index=0,
    )

# --- PAGE CONTENT ---
# The content of the main page changes based on the user's selection in the sidebar.

# --- HOME PAGE ---
if selected == "Home":
    st.title("Sharp Object Detection System")
    st.markdown("---")
    st.subheader("Welcome to the Real-Time Object Detection Application")
    st.markdown("""
        This application is a proof-of-concept for automated security surveillance. It uses a custom-trained YOLOv8 model to identify potentially dangerous objects from a live video feed.
        
        **How to Use:**
        1.  Navigate to the **Live Detection** page from the sidebar menu.
        2.  Grant permission for the app to access your webcam.
        3.  The application will start processing the feed and highlight any detected objects.
        
        This project demonstrates the power of computer vision for enhancing safety and security.
    """)
    st.image("https://miro.medium.com/1*Hr88QJB6DtH6C9dfsDrYWA.jpeg", caption="System Interface Preview")

# --- LIVE DETECTION PAGE ---
if selected == "Live Detection":
    st.header("🔴 Live Webcam Feed")
    st.markdown("The model is now running. Point your webcam at objects to see the detections.")

    # Confidence slider is now conditionally shown on this page in the sidebar
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.50, 0.05
    )
    st.sidebar.markdown("---")
    
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam. Please grant camera permissions and refresh.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame. The webcam may have been disconnected.")
                break

            # Run YOLO inference on the frame
            results = model(frame, verbose=False)

            # Draw boxes manually for full control over color and labels
            annotated_frame = frame.copy()
            for box in results[0].boxes:
                if box.conf[0] > confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    color = (0, 0, 255)  # Red color in BGR for all detections
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert the color from BGR (OpenCV's default) to RGB for Streamlit display
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(annotated_frame_rgb)
        
        # Release the camera resource when the loop is broken
        cap.release()

# --- ABOUT PAGE ---
if selected == "About":
    st.title("About This Project")
    st.markdown("---")
    st.info("""
        This application is a demonstration of a real-time object detection pipeline.
        
        **Core Technologies:**
        - **Streamlit:** For creating the interactive web user interface.
        - **OpenCV:** For handling the live webcam video feed.
        - **YOLOv8 (Ultralytics):** The deep learning model used for object detection.
        
        The model was custom-trained on a public dataset to specialize in identifying harmful objects. This mini project serves as a practical example of applying modern AI tools to solve real-world problems.
    """)