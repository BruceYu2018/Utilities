# -*- coding: UTF-8 -*-
import cv2, time
import numpy as np
import sys, ConfigParser, os, redis

redis_msg_transfer = redis.StrictRedis(host='localhost', port=6379, db=0)
redis_topic = "/WellOcean/vdmaster0"
video_save_dir = "/cv/westwell/save_videos"
video_shape = '1080'


def _get_image(redis_topic, resolution):
    redis_info = redis_msg_transfer.get(redis_topic)
    if redis_info is None:
        if resolution == '720':
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
        elif resolution == '1440':
            image = np.zeros((1440, 1024, 3), dtype=np.uint8)
        else:
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        return image
    img_buffer = np.frombuffer(redis_info, dtype=np.uint8)
    decode_img = cv2.imdecode(img_buffer, cv2.IMREAD_UNCHANGED)
    if decode_img.shape[0] == int(resolution):
        return decode_img
    if resolution == '720':
        image = cv2.resize(decode_img, (1280, 720))
    elif resolution == '1440':
        image = cv2.resize(decode_img, (1024, 1440))
    else:
        image = cv2.resize(decode_img, (1920, 1080))
    return image


if video_shape == '720':
    show_image = np.zeros(shape=(720, 1280, 3), dtype=np.uint8)
if video_shape == '1080':
    show_image = np.zeros(shape=(1080, 1920, 3), dtype=np.uint8)
height, width, channel = show_image.shape

cv_major_ver = int(cv2.__version__.split('.')[0])
if cv_major_ver == 2:
    fourcc = cv2.cv.CV_FOURCC('M', 'P', '4', '2')
elif cv_major_ver == 3:
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

timestamp = int(time.time()*1000)
video_name = str(timestamp) + '.avi'
video_path = os.path.join(video_save_dir, video_name)
output = cv2.VideoWriter(video_path, fourcc, 24.0, (1920, 1080))
frame_num = 0
while True:
    show_image = _get_image(redis_topic, video_shape)
    frame_num += 1
    if frame_num < 7500:
        output.write(show_image)
        time.sleep(0.04)
    else:
        timestamp = int(time.time() * 1000)
        video_name = str(timestamp) + '.avi'
        video_path = os.path.join(video_save_dir, video_name)
        output = cv2.VideoWriter(video_path, fourcc, 24.0, (1920, 1080))
        frame_num = 0
        print "save new video: " + str(video_path)
