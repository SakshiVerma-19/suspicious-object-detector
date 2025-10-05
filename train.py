from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8 model. 'yolov8n.pt' is the smallest and fastest.
    # For higher accuracy, you could use 'yolov8s.pt' or 'yolov8m.pt'.
    model = YOLO('runs/detect/train3/weights/last.pt') 

    # Train the model on your custom dataset
    # The 'data' argument should point to your data.yaml file.
    # 'epochs' determines how many times the model sees the entire dataset. 
    # For a demo, 25-50 epochs is a good starting point.
    # 'imgsz' is the image size the model will be trained on. 640 is common.
    print("Resuming training...")
    results = model.train(resume=True)
    print("Training complete!")
    print("Model and results saved in the 'runs' directory.")

if __name__ == '__main__':
    main()