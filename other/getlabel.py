import cv2 as cv
import os
import numpy as np

filepath = '/home/westwell/Downloads/ENet/Tailand_test'
train_list = os.listdir(os.path.join(filepath, 'train'))
train_label_list = os.listdir(os.path.join(filepath, 'train_label'))
train_list.sort()
train_label_list.sort()

if not os.path.exists(os.path.join(filepath, 'labelcheck')):
    os.mkdir(os.path.join(filepath, 'labelcheck'))

label_colours_bgr = np.zeros((1, 256, 3), np.uint8)
label_colours_bgr[0, 0] = [0, 0, 0]
label_colours_bgr[0, 1] = [0, 255, 0]
label_colours_bgr[0, 2] = [255, 0, 0]
label_colours_bgr[0, 3] = [0, 0, 255]
label_colours_bgr[0, 4] = [0, 255, 255]
label_colours_bgr[0, 5] = [255, 255, 0]
# label_colours_bgr[0, 6] = [205, 205, 0]

for img1, img2 in zip(train_list, train_label_list):
    srcImg = cv.imread(filepath + '/train/' + img1)
    labelImg = cv.imread(filepath + '/train_label/' + img2)
    coloredImg = np.zeros(labelImg.shape, dtype=np.uint8)
    cv.LUT(labelImg, label_colours_bgr, coloredImg)
    srcImg = cv.addWeighted(srcImg, 1, coloredImg, 0.2, 0)
    cv.imwrite(filepath + '/labelcheck/' + img1, srcImg)
    print 'processing ' + img1
