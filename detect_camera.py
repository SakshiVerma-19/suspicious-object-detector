import cv2
from ultralytics import YOLO

def main():
    # Load your custom-trained model
    # Make sure the path is correct
    model = YOLO('runs/detect/train3/weights/best.pt')

    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run inference on the current frame
        results = model(frame)

        # The 'results' object contains detection information.
        # .plot() is a handy method that draws the bounding boxes and labels on the frame.
        #annotated_frame = results[0].plot()
        annotated_frame = frame.copy() # Start with the original frame
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get the confidence score and class name
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            # Set the color to RED (in BGR format for OpenCV)
            color = (0, 0, 255)
            
            # Draw the rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create the label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Put the label text above the box
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the resulting frame
        cv2.imshow('Suspicious Object Detection', annotated_frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()