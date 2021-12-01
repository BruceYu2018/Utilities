import cv2
import numpy as np
import os

class labelCreator:

        def __init__(self, filepath, dirname):
            self.filepath = filepath
            self.dirname = dirname
            self.mask = 0
            self.label = 0
            if not os.path.exists(filepath + dirname):
                os.mkdir(filepath + dirname)

        def initialize(self, imgsize):
            self.mask = np.zeros(imgsize, np.uint8)
            self.label = np.zeros(imgsize, np.uint8)

        def createLabel(self, contour, labelcolor, allres, labelnum):
            cv2.drawContours(self.mask, [contour], 0, labelcolor, cv2.FILLED)
            cv2.drawContours(allres, [contour], 0, (labelnum, labelnum, labelnum), cv2.FILLED)

            return allres

        def writeLabel(self, imgname, srcImage, alpha):
            cv2.addWeighted(srcImage, 1, self.mask, alpha, 0, self.label)
            cv2.imwrite(self.filepath + self.dirname + '/' + imgname + '.png', self.label)