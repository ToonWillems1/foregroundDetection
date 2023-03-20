# Object detection code via OpenCV (https://www.youtube.com/watch?v=HXDD7-EnGBY)
import cv2
import datetime #can delete this import
import numpy as np
import argparse

# Set videocapture method
#cap = cv2.VideoCapture(0)  # to use laptop camera
input_video = '' #path to input video
cap = cv2.VideoCapture(input_video)  # to use input video

# Initialisation
radius = 2

classNames = []
classFile = 'labels.txt'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320) #frame size
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Loop for bounding box object detection
while cap.isOpened():
    success, img = cap.read()
    img = cv2.resize(img, (540, 450))
    #if time_in_range(start=datetime.time(8, 0, 0), end=datetime.time(18, 0, 0), current=datetime.datetime.now().time()):
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 5 < classId < 26:
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId-1].upper(), (box[0]+10,box[1]-20),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0))

    cv2.imshow("Output", img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
