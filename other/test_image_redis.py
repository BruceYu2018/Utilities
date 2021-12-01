import redis
import numpy as np
import cv2 as cv
import time

if __name__ == "__main__":
    # topic = '/WellOcean/osk2_master_shell_1/video_0'
    topic = '/tmp/master1'
    redis_msg_transfer = redis.StrictRedis(host='localhost', port=6379, db=0)
    while True:
        start = time.time()
        encode_img = redis_msg_transfer.get(topic)
        if encode_img is None:
            continue
        img_temp = np.frombuffer(encode_img, dtype=np.uint8)
        if img_temp is None:
            continue
        decode_img = cv.imdecode(img_temp, cv.IMREAD_UNCHANGED)
        end = time.time()
        print str((end - start)*1000) + " ms"
        cv.imshow("test", decode_img)
        cv.waitKey(1)
