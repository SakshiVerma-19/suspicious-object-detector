import streamlit as st
import cv2
from ultralytics import YOLO
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sharp Object Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    model = YOLO('runs/detect/train3/weights/best.pt')
    return model

model = load_model()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Live Detection", "About"],
        icons=["house-door-fill", "camera-video-fill", "info-circle-fill"],
        menu_icon="cast",
        default_index=0,
    )

# --- VIDEO PROCESSING CLASS ---
# This class will process each frame received from the browser
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.confidence_threshold = 0.5  # Default confidence

    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold

    def transform(self, frame):
        # Convert the frame to a NumPy array
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO inference
        results = model(img, verbose=False)

        # Draw boxes on the frame
        for box in results[0].boxes:
            if box.conf[0] > self.confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                color = (0, 0, 255)  # Red in BGR
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return img

# --- PAGE CONTENT ---

if selected == "Home":
    st.title("Sharp Object Detection System")
    st.markdown("---")
    st.subheader("Welcome to the Real-Time Object Detection Application")
    st.markdown("""
        This application uses a custom-trained YOLOv8 model to identify potentially dangerous objects from a live video feed.
        
        **How to Use:**
        1.  Navigate to the **Live Detection** page from the sidebar menu.
        2.  Click the **"START"** button and grant camera permissions.
        3.  The application will stream your webcam feed and highlight any detected objects.
    """)
    st.image("https://miro.medium.com/0*NKSu3VocwAoOPWyX.jpg", caption="System Interface Preview")

if selected == "Live Detection":
    st.header("🔴 Live Webcam Feed")
    st.markdown("Click the **START** button below to begin.")

    # Confidence slider in the sidebar
    conf_slider = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.50, 0.05
    )
    st.sidebar.markdown("---")
    
    # Use webrtc_streamer to handle the video stream
    ctx = webrtc_streamer(
        key="detection",
        video_processor_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    # Update the confidence threshold in the video transformer
    if ctx.video_processor:
        ctx.video_processor.set_confidence_threshold(conf_slider)

if selected == "About":
    st.title("About This Project")
    # ... (rest of the About page code) ...