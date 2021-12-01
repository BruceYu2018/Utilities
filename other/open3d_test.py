import os
import math
import numpy as np
from open3d import *

FOR1 = create_mesh_coordinate_frame(size=0.5, origin=[0, 0, 0])

degree = math.pi / 2
rotateMatrix = np.array([[round(math.cos(degree), 4), 0, round(-math.sin(degree), 4)],
                         [0, 1, 0],
                         [round(math.sin(degree), 4), 0, round(math.cos(degree), 4)]])
print rotateMatrix
points = np.random.rand(10000, 3) + 1
# points = np.dot(points, rotateMatrix)
point_cloud = PointCloud()
point_cloud.points = Vector3dVector(points)
draw_geometries([FOR1, point_cloud])