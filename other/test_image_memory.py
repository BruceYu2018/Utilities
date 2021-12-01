import numpy as np
import mmap
import cv2 as cv
import fcntl
import time

if __name__ == '__main__':
    height = 1080
    width = 1920

    mmap_name = '/tmp/master1'
    share_file = open(mmap_name, 'r')
    shmmap = mmap.mmap(share_file.fileno(), 0, prot=mmap.PROT_READ)

    while True:
        start = time.time()
        fcntl.flock(share_file, fcntl.LOCK_EX)
        shmmap.seek(0)
        string = shmmap.read(width*height*3)
        fcntl.flock(share_file, fcntl.LOCK_UN)
        image_str = np.fromstring(string, dtype=np.uint8)
        image = np.reshape(image_str, (height, width, 3))
        end = time.time()
        print str((end - start)*1000) + " ms"
        cv.imshow('test', image)
        cv.waitKey(1)

    shmmap.close()
