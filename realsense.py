import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Model for object detection
model = YOLO('yolo11n.pt')
names = model.names

# Confidence threshold for prediction
threshold = 0.25

#blur ratio
blur_ratio = 1


def main():
    # Open the camera (usually the default camera is 0)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Run detect model over the frame
        results = model.predict(frame, conf=threshold)

        boxes = results[0].boxes.xyxy.cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        annotator = Annotator(frame, line_width=2, example=names)

        #reset operator counter since its a new frame
        operator_count = 0

        #check if a detect occured
        if boxes is not None:
            for box, cls in zip(boxes, clss):
                #only look at class 0 (person class)
                #if int(cls) == 0:
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    obj = frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                    blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))

                    frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj

                    operator_count += 1  # Increment count for each person detected


            print(f'COUNT-----{operator_count}')

        # Display the resulting frame
        cv2.imshow('Camera Feed', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
