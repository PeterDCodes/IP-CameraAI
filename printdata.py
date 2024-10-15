import cv2
from ultralytics import YOLO


#global variabales for Data---------------------------
site = 'redink'
area = 'gamelab'
line = 'rp'
camera = '1'
zone = 'station1'
timestamp = 0
status = 0


#JSON Str Format--------------------------------------
data = f'{{"site":"{site}", "area":"{area}", "line":"{line}", "camera":"{camera}", "zone":"{zone}", "timestamp":"{timestamp}", "status":"{status}"}}'


# Model for object detection---------------------------
try:
    model = YOLO('yolo11n.pt')
except Exception as e:
    print('Model Not Found')
    exit()

#Model confidence threshold
THRESHOLD = 0.5


#select webcam
video_feed = cv2.VideoCapture(0)
#run video frame by frame
while True:
    ret, frame = video_feed.read()

    if not ret:
        print('Error Reading Frame')
        break

    results=model.predict(frame, conf = THRESHOLD, save_txt = True)

    annotated_frame = results[0].plot()

    # Display the captured frame
    cv2.imshow('Camera', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
video_feed.release()
cv2.destroyAllWindows()

#if person detected set status to 1
#if change in status, print data
