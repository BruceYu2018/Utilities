#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2, time, os
import numpy as np

video_source_path = "/cv/qingdao/test_fengche/IPXU3231277.avi"
black_image = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)

video_save_path = '/cv/qingdao/test_fengche/'
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

second_for_black = 45
frame_num = 0
while True:
    if frame_num == 0:
        time1 = str(int(time.time()))
        output = cv2.VideoWriter(video_save_path + time1 + '.avi', fourcc, FPS, (1920, 1080))
    if frame_num < second_for_black * frame_rate:
        output.write(black_image)
        show_image = black_image
    else:
        ret, image = cap.read()
        if not ret:
            cap.release()
            break
        output.write(image)
        show_image = image

    cv2.imshow("test", show_image)
    cv2.waitKey(1)
    frame_num += 1
