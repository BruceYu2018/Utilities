from PIL import Image
from pylab import *
import cv2,time
import ctypes as C

im = array(Image.open('/home/westwell/Downloads/transform/test.jpg'))
# imshow(im)
# print 'Please click 8 points'
# x = ginput(4,-1)
# print 'you clicked:', x


# srcPoint=np.array(x[0:4],np.float32)
# dstPoint=np.array(x[4:],np.float32)

shangyi = 150
xyasuo = -150
srcPoint=np.array([[415,565],[848,594],[833,750],[417,710]],np.float32)
dstPoint=np.array([[415 - xyasuo,565-shangyi],[848 - xyasuo,565-shangyi],[848 - xyasuo,710-shangyi],[415 - xyasuo,710-shangyi]],np.float32)
#
persepectiveMatrix=cv2.getPerspectiveTransform(srcPoint,dstPoint)
# reversePersepectiveMatrix=cv2.getPerspectiveTransform(dstPoint,srcPoint)
np.save('/home/westwell/Downloads/transform/perspective_0.npy',persepectiveMatrix)
# np.save('/cv2/daxiezhakou/trigger/reverseperspective_0.npy',reversePersepectiveMatrix)
# persepectiveMatrix = np.load('/home/westwell/Desktop/controlright_1/pers/perspective_0.npy')
# rotateMatrix = cv2.getRotationMatrix2D(((1920 - 1) / 2.0, (1080 - 1) / 2.0), -90, 1)
# add_row = np.array([0, 0, 1])
# rotateMatrix = np.row_stack((rotateMatrix, add_row))
# print rotateMatrix
# persepectiveMatrix = np.dot(persepectiveMatrix, rotateMatrix)
# persepectiveMatrix = np.dot(persepectiveMatrix, translateMatrix)
dstimage=cv2.warpPerspective(im,persepectiveMatrix,(1920,1080))
# rotateMatrix = cv2.getRotationMatrix2D(((1920 - 1) / 2.0, (1080 - 1) / 2.0), -90, 1)
# dstimage = cv2.warpPerspective(im, persepectiveMatrix, (1920,1080))
# cv2.imwrite('/cv/WellOcean_Projects_Save/doorlift_crane/launcherfile_real/gct2/webwxgetmsgimgleft_pers.jpeg',dstimage)
# src_pered=cv2.warpPerspective(dstimage,reversePersepectiveMatrix,(1920,1080))
cv2.imshow('src',im)
cv2.imshow('dst1',dstimage)
# cv2.imshow('dst2',src_pered)
cv2.waitKey(0)