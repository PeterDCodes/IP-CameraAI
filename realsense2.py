import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use the desired model, e.g., 'yolov8n.pt' for YOLOv8 Nano

    # Create a pipeline
    pipeline = rs.pipeline()

    # Configure the pipeline to stream color and depth
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the pipeline
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert the color frame to a numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Run YOLOv8 inference
            results = model(color_image)

            # Process results
            for result in results:
                boxes = result.boxes  # Bounding boxes
                for box in boxes:
                    # Get the coordinates and confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the box coordinates
                    conf = box.conf[0]  # Confidence

                    # Draw the bounding box and label on the frame
                    label = f'Class: {int(box.cls[0])}, Conf: {conf:.2f}'
                    color_image = cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Display the frame with detections
            cv2.imshow('RealSense Color Frame with YOLOv8', color_image)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the pipeline
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
