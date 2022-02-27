import cv2
import numpy as np

def rotateimg(img, angle):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
    image = cv2.warpAffine(img, M, (cols, rows))
    return image

# test_image = cv2.imread("/home/westwell/Pictures/image.png")
test_image = np.zeros((5, 0, 3), np.uint8)
test_image = rotateimg(test_image, 180)
cv2.imshow("test", test_image)
cv2.waitKey(0)