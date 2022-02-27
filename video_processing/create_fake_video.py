#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import time
import os
import numpy as np

video_1 = "/cv/qingdao/1570519631.avi"
video_2 = "/cv/qingdao/1570520505.avi"
mid_line = 700

video_save_path = '/cv/qingdao/'
if not os.path.exists(video_save_path):
    os.makedirs(video_save_path)

cap_1 = cv2.VideoCapture(video_1)
cap_2 = cv2.VideoCapture(video_2)
frame_rate = cap_1.get(cv2.CAP_PROP_FPS)

FPS = frame_rate
cv_major_ver = int(cv2.__version__.split('.')[0])
if cv_major_ver == 2:
    fourcc = cv2.cv.CV_FOURCC('M', 'P', '4', '2')
elif cv_major_ver == 3:
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')


frame_num = 0
ret_2 = True
while True:
    ret_1, image = cap_1.read()
    if not ret_1:
        cap_1.release()
        ret_2, image = cap_2.read()
    if not ret_2:
        cap_2.release()
        break
    if frame_num == 0:
        time1 = str(int(time.time()))
        output = cv2.VideoWriter(video_save_path + time1 + '.avi', fourcc, FPS, (1920, 1080))
    output.write(image)
    show_image = image
    cv2.imshow("test", show_image)
    cv2.waitKey(1)
    frame_num += 1

# frame_num = 0
# while True:
#     ret_1, image_1 = cap_1.read()
#     ret_2, image_2 = cap_2.read()
#     if not ret_1 or not ret_2:
#         cap_1.release()
#         cap_2.release()
#         break
#     if frame_num == 0:
#         time1 = str(int(time.time()))
#         output = cv2.VideoWriter(video_save_path + time1 + '.avi', fourcc, FPS, (1920, 1080))
#     created_image = np.concatenate((image_1[:, :mid_line, :], image_2[:, mid_line:, :]), 1)
#     output.write(created_image)
#     show_image = created_image
#     cv2.imshow("test", show_image)
#     cv2.waitKey(1)
#     frame_num += 1