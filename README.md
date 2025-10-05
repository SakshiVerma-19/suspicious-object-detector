Real-Time Suspicious Object Detection
A Streamlit web application that uses a custom-trained YOLOv8 model to detect sharp objects like knives and scissors in real-time from a webcam feed.
Be sure to replace interface_preview.png with a screenshot of your own running application!

## About The Project
This project is a proof-of-concept for an automated security surveillance system. It addresses the challenge of manually monitoring video feeds by leveraging computer vision to automatically identify and flag potentially dangerous objects. The core of the application is a fine-tuned YOLOv8 model that processes a live webcam stream, highlighting any detected threats for immediate attention.

### Key Features:
Real-time Detection: Analyzes video frames from a live webcam feed with minimal latency.
Custom Model: Utilizes a YOLOv8n model fine-tuned specifically on a dataset of sharp objects.
Interactive UI: A clean, multi-page user interface built with Streamlit for easy navigation and control.
Visual Feedback: Overlays red bounding boxes and confidence scores on detected objects directly in the video stream.

## Built With
This project was built using the following technologies:
Python
Streamlit
YOLOv8 (Ultralytics)
OpenCV
PyTorch

## Getting Started
To get a local copy up and running, follow these simple steps.

### Prerequisites
Make sure you have Python 3.9+ and pip installed on your system.

### Installation
Clone the repository

git clone https://github.com/your_username/your_repository_name.git
Navigate to the project directory

cd your_repository_name
Create and activate a virtual environment

On Windows:
python -m venv venv
venv\Scripts\activate
On macOS/Linux:

python3 -m venv venv
source venv/bin/activate
Install the required packages
pip install -r requirements.txt

## Usage
Once the installation is complete, you can run the application with a single command:
streamlit run app.py
Your web browser will open, and you can navigate to the "Live Detection" page from the sidebar to start the webcam feed. You may need to grant camera permissions in your browser.

## License
Distributed under the MIT License. See LICENSE for more information.
