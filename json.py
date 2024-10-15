import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import threading


#global variabales for Data---------------------------
site = 'redink'
area = 'gamelab'
line = 'rp'
camera = '1'
zone = 'station1'
timestamp = 0
status = 0


#JSON Str Format--------------------------------------
data = f'{
    "site":{site},
    "area":{area},
    "line":{line},
    "camera":{camera},
    "zone":{zone},
    "timestamp":{timestamp}
    "status":{status}
}'

# Model for object detection---------------------------
model = YOLO('yolo11n.pt')


#run webcam
#if person detected set status to 1
#if change in status, print data
