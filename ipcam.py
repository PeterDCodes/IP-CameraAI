import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import threading

# Model for object detection
model = YOLO('yolo11n.pt')
names = model.names

# Confidence threshold for prediction
threshold = 0.50

#blur ratio
blur_ratio = 200

# Example RTSP URL for the main stream
ip_camera_url = 'rtsp://192.168.226.201:554/profile1'

# Desired frame width and height (resize input frame)
frame_width = 640
frame_height = 480

# Threading for video capture
class VideoCaptureThread(threading.Thread):
    def __init__(self, url):
        super().__init__()
        self.cap = cv2.VideoCapture(url)
        self.frame = None
        self.ret = False
        self.running = True

    def run(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                print("Failed to grab frame")
                break

    def stop(self):
        self.running = False
        self.cap.release()

capture_thread = VideoCaptureThread(ip_camera_url)
capture_thread.start()

#frame count used to limit number of detects
frame_count = 0

while True:
    if capture_thread.ret:
        frame = capture_thread.frame
        frame_count += 1

        #will run every 50x frames to reduce frequency?????// does this work? maybe a better way using time intervals to calc
        if True:#frame_count % 50 == 0:
            # Resize frame before inference for faster processing
            frame = cv2.resize(frame, (frame_width, frame_height))

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
                    if int(cls) == 0:
                        annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                        obj = frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]
                        blur_obj = cv2.blur(obj, (blur_ratio, blur_ratio))

                        frame[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])] = blur_obj

                        operator_count += 1  # Increment count for each person detected


            print(f'COUNT-----{operator_count}')
            # Display the frame
            cv2.imshow('IP Camera Stream', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_thread.stop()
cv2.destroyAllWindows()


#need to write in the total count of people
#print number of objects detected
#then write that viat HTTP->node-red->PostgreSQL
