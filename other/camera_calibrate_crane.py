# import numpy as np
# import cv2
# import glob
# import matplotlib.pyplot as plt
#
# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6 * 8, 3), np.float32)
# objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
#
# # Arrays to store object points and image points from all the images.
# objpoints = []  # 3d points in real world space
# imgpoints = []  # 2d points in image plane.
#
# # Make a list of calibration images
# # images = glob.glob('calibration_wide/GO*.jpg')
#
# # Step through the list and search for chessboard corners
# cap = cv2.VideoCapture(-1)
# # cap = cv2.VideoCapture("rtsp://admin:12345@192.168.101.207/main/Channels/1")
# while cap.isOpened():
#     success, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
#
#     if ret == True:
#         cv2.drawChessboardCorners(img, (11, 8), corners, ret)
#     cv2.imshow('img', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
# cap.release()

# cv2.destroyAllWindows()
#
# import pickle
#
# # Test undistortion on an image
# img = cv2.imread('calibration_wide/test_image.jpg')
# img_size = (img.shape[1], img.shape[0])
#
# # Do camera calibration given object points and image points
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
#
# dst = cv2.undistort(img, mtx, dist, None, mtx)
# cv2.imwrite('calibration_wide/test_undist.jpg', dst)
#
# # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
# dist_pickle = {}
# dist_pickle["mtx"] = mtx
# dist_pickle["dist"] = dist
# pickle.dump(dist_pickle, open("calibration_wide/wide_dist_pickle.p", "wb"))
# # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# # Visualize undistortion
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=30)
# ax2.imshow(dst)
# ax2.set_title('Undistorted Image', fontsize=30)


import cv2
import queue
import time
import threading
import numpy as np

# q = queue.Queue()



def Receive():
    print("start Reveive")
    cap = cv2.VideoCapture("rtsp://admin:12345@192.168.101.207")
    global image, condition
    ret, frame = cap.read()
    # q.put(frame)
    while ret:
        ret, frame = cap.read()
        image = frame.copy()
        condition = ret
        cv2.imshow("frame1", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # q.put(frame)


def Display():
    print("Start Displaying")
    global image, condition
    while True:
        if condition:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                cv2.drawChessboardCorners(image, (11, 8), corners, ret)
                points = cv2.boxPoints(cv2.minAreaRect(corners)).astype('int')
                points = map(tuple, points)
                final_points = find_points(corners, points)
                for index in range(4):
                    cv2.line(image, final_points[index], final_points[(index+1)%4], (255,255,0), 2)
                srcPoint = np.array(final_points).astype('float32')
                dstPoint = np.array([[960, 560], [320, 560], [320, 160], [960, 160]]).astype('float32')
                persepectiveMatrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)
                dstimage = cv2.warpPerspective(image, persepectiveMatrix, (1920, 1080))
                cv2.imshow("frame2", dstimage)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

def find_points(corner_points, rectangle_points):
    final_points = []
    corner_points = corner_points.astype('int')[:,0,:]
    for each_point in rectangle_points:
        dist_matrix = np.zeros((corner_points.shape[0]))
        for index, corner_point in enumerate(corner_points):
            dist_matrix[index] = np.sqrt(np.sum(np.square(corner_point - each_point)))
        min_index = np.argmin(dist_matrix)
        final_points.append(tuple(corner_points[min_index].tolist()))
        np.delete(corner_points, min_index, 0)
    return final_points



if __name__ == '__main__':
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    condition = False
    p1 = threading.Thread(target=Receive)
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()

