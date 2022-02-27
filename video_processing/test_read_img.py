import redis
import time
import json
import cv2
import threading


def send_image():
    count = 0
    while count < 6:
        timestamp = int(time.time() * 1000)
        raw_img = cv2.imread('/home/westwell/Downloads/Data/sample_images/zhu_image/160629210004500Q20 front.jpg')
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        retval, encode_img = cv2.imencode('.jpg', raw_img, encode_param)
        encode_img_string = encode_img.tostring() + str(timestamp)
        redis_msg_transfer.hset(name='aerial_TT10000003', key='left_image1', value=encode_img_string)
        print str(int(time.time() * 1000) - int(timestamp)) + " ms"
        count = count + 1
    print "finish send image"


if __name__ == '__main__':
    redis_msg_transfer = redis.StrictRedis(host='192.168.2.11', port=6379, db=0)
    while True:
        timestamp = int(time.time() * 1000)
        send_image()
        print str(int(time.time() * 1000) - int(timestamp)) + " ms"
