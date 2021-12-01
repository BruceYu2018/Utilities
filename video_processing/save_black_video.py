#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2,time,sys,os
import numpy as np
import redis

redis_msg_transfer = redis.StrictRedis(host='localhost', port=6379, db=0)

showimage = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)
blackimage = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)
height, width, channel = showimage.shape

videosavepath='/cv/westwell/original_video/'
if not os.path.exists(videosavepath):
    os.makedirs(videosavepath)

FPS=12.0
cv_major_ver = int(cv2.__version__.split('.')[0])
if cv_major_ver == 2:
    fourcc = cv2.cv.CV_FOURCC('M', 'P', '4', '2')
elif cv_major_ver == 3:
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
frame_num = 0
while True:
    if frame_num == 0:
       time1=str(int(time.time())) + "_black_video"
       output=cv2.VideoWriter(videosavepath+time1+'.avi',fourcc,FPS,(1920,1080))
    redis_info = redis_msg_transfer.get('test')
    if redis_info is None:
        showimage = blackimage
    else:
        img_buffer = np.frombuffer(redis_info, dtype=np.uint8)
        decode_img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
        showimage = decode_img
    # cv2.imshow('westwell', showimage)
    # cv2.waitKey(1)
    frame_num +=1
    if frame_num<20000:
       output.write(showimage)
       time.sleep(0.04)
    else:
       frame_num = 0




