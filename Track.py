#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""""
Implement DeepSORT using YOLOv5 (any custom model) as detector.
@author: STRH  
"""

import os
import cv2
import torch
import numpy as np
import time 
import glob
import sys
from deep_sort_realtime.deepsort_tracker import DeepSort

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class YOLOv5_Detect():
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self, model_name):
        if model_name: 
            model = torch.hub.load('ultralytics/yolov5', 'custom', path= model_name, force_reload= True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    

    def score_frame(self, frame):
        self.model.to(self.device)
        downscale_factor = 2
        height, width = frame.shape[0], frame.shape[1]
        width = int(width/downscale_factor)
        height = int(height/downscale_factor)
        frame = cv2.resize(frame, (width, height))
        results = self.model(frame)

        labels, coord = results.xyxyn[0][:,-1], results.xyxyn[0][:, :-1]
        return labels, coord
    
    # def class_to_label(self, x):
    #     return self.classes(int(x))
    
    def draw_bbox(self, results,  frame,  height, width,  conf=0.1):
        labels, coord = results
        detections = []

        n = len(labels)
        x_shape, y_shape = width,  height

        for i in range(n):
            row = coord[i]

            if row[4] >= conf:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                # if self.class_to_label(labels[i]) == 'person':
                #     x_center = x1 + (x2 - x1)
                #     y_center = y1 + ((y2 - y1)/2)

                #     tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                conf = float(row[4])
                    # feature = 'person'

                detections.append(([x1, y1, int(x2-x1), int(y2-y1)], row[4], 'person'))

        return frame, detections
    
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector =  YOLOv5_Detect(model_name=None)
object_tracker = DeepSort(max_age= 5, n_init= 2, nms_max_overlap= 1.0, max_cosine_distance= 0.3, nn_budget= None, override_track_class= None, 
                          embedder= "mobilenet", half = True, bgr = True, embedder_gpu= True, embedder_model_name= None, embedder_wts= None,
                          polygon= False, today= None)

while cap.isOpened():
    ret, frame = cap.read()
    start_time = time.time()

    results = detector.score_frame(frame)
    img,  detections = detector.draw_bbox(results, frame, height=frame.shape[0], width=frame.shape[1], conf=0.1)
    tracks = object_tracker.update_tracks(detections, frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb

        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255),2)
        cv2.putText(frame, f"ID: {str(track_id)}", (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    end = time.time()
    total_time = end - start_time
    fps = 1/total_time

    cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()