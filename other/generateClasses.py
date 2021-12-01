import json
import cv2
import numpy as np
import os

filepath = '/home/zhenyu/Downloads/test'
filelist = os.listdir(filepath)
filelist.sort()

if not os.path.exists(filepath + '/container'):
   os.mkdir(filepath + '/container')
if not os.path.exists(filepath + '/train_labels'):
   os.mkdir(filepath + '/train_labels')
if not os.path.exists(filepath + '/train'):
   os.mkdir(filepath + '/train')

count = 1
print 'loading ...'
for file in filelist:
    filename = os.path.splitext(file)
    if filename[1] == '.png':
       print 'processing image ' + str(count)
       srcImage = cv2.imread(filepath + '/' + file)
    if filename[1] == '.json':
       with open(filepath + '/' + file, 'r') as jsonfile:
            data = json.load(jsonfile)
       containerImg = np.zeros((data['imgHeight'], data['imgWidth'], 3), np.uint8)
       res = np.zeros((data['imgHeight'], data['imgWidth'], 3), np.uint8)

       for object in data['objects']:
           if object['deleted'] == 0:
              contour = np.array(object['polygon']).astype(np.int32)
              if object['label'] == 'road':
                 cv2.drawContours(containerImg, [contour], 0, (255, 255, 255), cv2.FILLED)
                 cv2.drawContours(res, [contour], 0, (1, 1, 1), cv2.FILLED)
                 #cv2.addWeighted(srcImage, 1, containerImg, 0.5, 0, containerImg)
                 imgname = str(100000 + count)
                 cv2.imwrite(filepath + '/train' + '/' + imgname[1:] + '.png', srcImage)
                 cv2.imwrite(filepath + '/container' + '/' + imgname[1:] + '.png', containerImg)
                 cv2.imwrite(filepath + '/train_labels' + '/' + imgname[1:] + '.png', res)
       count += 1