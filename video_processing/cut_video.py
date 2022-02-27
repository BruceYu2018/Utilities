#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import time
import os

video_source_path = "/cv/qingdao/01000002253000000.mp4"
cut_start = 4.5  # minute 9
cut_duration = 2  # minute 0.86

video_save_path = '/cv/qingdao/'
if not os.path.exists(video_save_path):
    os.makedirs(video_save_path)

cap = cv2.VideoCapture(video_source_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)

FPS = frame_rate
cv_major_ver = int(cv2.__version__.split('.')[0])
if cv_major_ver == 2:
    fourcc = cv2.cv.CV_FOURCC('M', 'P', '4', '2')
elif cv_major_ver == 3:
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

frame_num = 0
while True:
    ret, image = cap.read()
    if not ret:
        cap.release()
        break
    if frame_num == 0:
        time1 = str(int(time.time()))
        output = cv2.VideoWriter(video_save_path + time1 + '.avi', fourcc, FPS, (1920, 1080))
    if frame_num < cut_start * 60 * frame_rate:
        print "cut frame " + str(frame_num)
        frame_num += 1
        continue
    elif frame_num > (cut_start + cut_duration) * 60 * frame_rate:
        # if frame_num == int((cut_start + cut_duration) * 60 * frame_rate) + 1:
        #     static_image = image
        # output.write(static_image)
        # show_image = static_image
        # cv2.imshow("test", show_image)
        # cv2.waitKey(1)
        print "cut frame " + str(frame_num)
        frame_num += 1
        continue
    else:
        output.write(image)
        show_image = image
        cv2.imshow("test", show_image)
        cv2.waitKey(1)
    frame_num += 1