# -*- coding: UTF-8 -*-

# 给定一个3D点的list，画OBB(oriented bounding-box)
import numpy as np
import matplotlib.pyplot as plt
import math
# do not delete this
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import time
import os
import open3d as o3d

class_list = ["spreader", "container", "truckhead", "tray"]


def get_points_box(x, y, z, l, w, h):
    x1 = [x - w / 2, y - h / 2, z - l / 2]
    x2 = [x - w / 2, y - h / 2, z + l / 2]

    x3 = [x - w / 2, y + h / 2, z - l / 2]
    x4 = [x - w / 2, y + h / 2, z + l / 2]

    x5 = [x + w / 2, y - h / 2, z - l / 2]
    x6 = [x + w / 2, y - h / 2, z + l / 2]

    x7 = [x + w / 2, y + h / 2, z - l / 2]
    x8 = [x + w / 2, y + h / 2, z + l / 2]
    points_box = [x2, x1, x3, x4, x6, x5, x7, x8]
    return points_box


def custom_draw_geometry(pcd, linesets):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(linesets)
    render_option = vis.get_render_option()
    render_option.point_size = 1
    render_option.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()


def read_points(file):
    f = open(file, 'r')
    lines = f.readlines()
    l = len(lines)
    points = np.zeros((l, 4), np.float32)

    for i in range(l):
        aa = lines[i].split(' ')
        points[i, 0] = float(aa[0])
        points[i, 1] = float(aa[1])
        points[i, 2] = float(aa[2])
        points[i, 3] = float(aa[3])

    f.close()
    return points


def computer_distance_horizontal(plane, points):
    a, b = plane[:2]
    points_l = filter(lambda point: np.dot((a, b), (point[0], point[1])) <= 1, points)
    points_r = filter(lambda point: np.dot((a, b), (point[0], point[1])) > 1, points)
    dist_l = map(lambda point: abs(np.dot((a, b, -1), (point[0], point[1], 1))), points_l)
    dist_r = map(lambda point: abs(np.dot((a, b, -1), (point[0], point[1], 1))), points_r)
    dist = (np.max(dist_l) + np.max(dist_r)) / np.linalg.norm(plane[:3])
    return dist


def computer_distance_vertical(z, points):
    points_u = filter(lambda point: point[2] > z, points)
    points_d = filter(lambda point: point[2] <= z, points)
    dist_u = map(lambda point: abs(z - point[2]), points_u)
    dist_d = map(lambda point: abs(z - point[2]), points_d)
    # print '------'
    # print 'z', z
    # print points[:, 2]
    # print 'point_u'
    # print points_u
    # print 'points_d'
    # print points_d
    # print 'dist_u', dist_u
    # print 'dist_d', dist_d
    dist = np.max(dist_u) + np.max(dist_d)
    return dist


def computer_bbox_3d_pca(points):
    x1, y1, z1 = np.mean(points[:, :3], axis=0)
    pca = PCA()
    pca.fit(points[:, :2])
    p1, p2 = pca.components_
    p1 = np.append(p1, 0)
    p2 = np.append(p2, 0)
    p3 = (0, 0, 1)

    a1, b1, c1 = p1
    y_rotation = np.arccos(np.dot(p1, (0, -1, 0)))
    a = 1.0 / (x1 - a1 * y1 / b1)
    b = -a * a1 / b1
    width = computer_distance_horizontal((a, b, 0, -1), points)

    a1, b1, c1 = p2
    a = 1.0 / (x1 - a1 * y1 / b1)
    b = -a * a1 / b1
    length = computer_distance_horizontal((a, b, 0, -1), points)

    height = computer_distance_vertical(z1, points)

    return width, height, length, -y1, -z1, x1, y_rotation


# Project edge points on angle function
def project_edge_points_on_angle(points, theta):
    # Define edge vectors
    l1 = np.asarray([np.cos(theta), np.sin(theta)])
    l2 = np.asarray([-np.sin(theta), np.cos(theta)])
    # Iterate over points in single points and calc projections c along l
    # c1, c2 = [], []
    # for point in points:
    # Add calculated projections to arrays
    # c1.append(np.asarray([point[0] * l1[0], point[1] * l1[1]]))
    # c2.append(np.asarray([point[0] * l2[0], point[1] * l2[1]]))
    # Calc projection via matrix multiplication/scalar product
    c1 = np.matmul(np.asarray(points), l1)
    c2 = np.matmul(np.asarray(points), l2)
    # Return projections
    return [c1, c2]


# Calculate closeness (criterion) function
def calculate_closeness(c1, c2, minimum_distance):
    # Get max projection from both sets
    c1_max, c2_max = np.amax(c1), np.amax(c2)
    c1_min, c2_min = np.amin(c1), np.amin(c2)
    # Init empty index arrays
    d1_max, d1_min, d2_max, d2_min, d1, d2 = [], [], [], [], [], []
    for c1_val, c2_val in zip(c1, c2):
        # Calculate distance vectors containing all distances between the boundaries and each point
        d1_max.append(c1_max - c1_val)
        d1_min.append(c1_val - c1_min)
        d2_max.append(c2_max - c2_val)
        d2_min.append(c2_val - c2_min)
    # Choose distance vector with smaller overall distances
    if np.linalg.norm(d1_max) > np.linalg.norm(d1_min):
        d1 = d1_min
    else:
        d1 = d1_max
    if np.linalg.norm(d2_max) > np.linalg.norm(d2_min):
        d2 = d2_min
    else:
        d2 = d2_max
    # Init with zero quality
    quality = 0
    for n in range(len(d1)):
        # Choose smallest distance from d1, d2
        d = np.amax([np.amin([d1[n], d2[n]]), minimum_distance])
        # Increase quality, quality increases faster if distances d are small
        quality = quality + 1 / d
    # Return quality
    return quality


# Calculate intersection point function
def calc_intersection_point(a1, b1, c1, a2, b2, c2):
    if np.abs(b1) < 0.01:
        b1 = -0.01
    if np.abs(b2) < 0.01:
        b2 = -0.01
    x = (c1 / b1 - c2 / b2) * np.power(a1 / b1 - a2 / b2, -1)
    y = (c1 - a1 * x) / b1
    return np.asarray([x, y])


# Search-based rectangle fitting function
def search_rectangle_fit(points, delta):
    # Init array for calculated qualities
    qualities = []
    # Workaround: Range function does not work with floats
    # Calc number of steps between 0 and 90 deg. with chosen step width
    steps_theta = int(round(np.pi / (2 * delta)))
    # Iterate over all steps, choose based on step number
    for num_theta in range(steps_theta + 1):
        # Choose
        theta = np.amin([num_theta * delta, np.pi / 2])
        c1, c2 = project_edge_points_on_angle(points, theta)
        # Calc quality of fit considering closeness as criterion
        quality = calculate_closeness(c1, c2, 0.01)
        qualities.append(np.asarray([quality, theta]))
    # Find angle with highest quality of fit
    theta_max = np.transpose(np.asarray(qualities))[1][np.argmax(np.transpose(np.asarray(qualities))[0])]
    # Get related projection
    c1_max, c2_max = project_edge_points_on_angle(points, theta_max)
    # Get rectangle parameters based on highest-quality angle
    c1, c2 = np.amin(c1_max), np.amin(c2_max)
    c3, c4 = np.amax(c1_max), np.amax(c2_max)
    # Rectangle parameter a1, b2, a3, b4
    v1 = np.cos(theta_max)
    # Rectangle parameter b1, b3
    v2 = np.sin(theta_max)
    # Rectangle parameter a2, a4
    v3 = -np.sin(theta_max)
    p1 = calc_intersection_point(v1, v2, c1, v3, v1, c2)
    p2 = calc_intersection_point(v3, v1, c2, v1, v2, c3)
    p3 = calc_intersection_point(v1, v2, c3, v3, v1, c4)
    p4 = calc_intersection_point(v3, v1, c4, v1, v2, c1)
    # print("Rectangle fit:", p1, p2, p3, p4)
    return [p1, p2, p3, p4]


def computer_bbox_3d_l_shape_fitting(points):
    p1, p2, p3, p4 = search_rectangle_fit(points[:, :2], 0.02)
    a = p2 - p1
    y_rotation = np.pi / 2 - np.arctan(a[1] / a[0])
    length = np.linalg.norm(a)
    b = p3 - p2
    width = np.linalg.norm(b)
    x1, y1 = (p1 + p3) / 2
    height = np.max(points[:, 2]) - np.min(points[:, 2])
    return width, height, length, -y1, 0.8, x1, y_rotation


def get_xyz_lwh(point_clouds):
    max_xyz = np.max(point_clouds, axis=0)
    min_xyz = np.min(point_clouds, axis=0)
    l = abs(max_xyz[0] - min_xyz[0])
    w = abs(max_xyz[1] - min_xyz[1])
    h = abs(max_xyz[2] - min_xyz[2])

    x = (max_xyz[0] + min_xyz[0]) / 2
    y = (max_xyz[1] + min_xyz[1]) / 2
    z = (max_xyz[2] + min_xyz[2]) / 2

    return x, y, z, l, w, h


def process_a_frame(path, index, bin_save_dir, label_save_dir):
    index = str(index).zfill(6)
    files = os.listdir(path)
    files = filter(lambda x: 'asc' in x, files)
    bin_file_name = os.path.join(bin_save_dir, index + '.bin')
    label_f = open(os.path.join(label_save_dir, index + '.txt'), 'w+')
    all_points = np.zeros((0, 4), dtype=np.float32)
    for file_name in files:
        points = read_points(os.path.join(path, file_name))
        all_points = np.concatenate((points, all_points))

        if 'background' not in file_name:
            x, y, z, l, w, h = get_xyz_lwh(points)

            y_rotation = 0
            class_name = file_name.split("_")[0]
            outstring = class_name + " " + \
                        "0.00 " + \
                        "0 " + \
                        "-10 " + \
                        "-1 " + \
                        "-1 " + \
                        "-1 " + \
                        "-1 " + \
                        str(round(h, 2)) + " " + \
                        str(round(w, 2)) + " " + \
                        str(round(l, 2)) + " " + \
                        str(round(x, 2)) + " " + \
                        str(round(y, 2)) + " " + \
                        str(round(z, 2)) + " " + \
                        str(y_rotation) + "\n"
            label_f.write(outstring)
    label_f.close()
    all_points.flatten()
    all_points.tofile(bin_file_name)


if __name__ == '__main__':
    bin_save_dir = '/cv/DataSet/LidarData/TrainDataAvia/training/velodyne'
    label_save_dir = '/cv/DataSet/LidarData/TrainDataAvia/training/label_2'
    original_label_dir = '/cv/DataSet/LidarData/RawData/point/pcd_datasets_result/lcb_pcd_result'
    all_paths = os.listdir(original_label_dir)
    all_paths.sort()
    count = 0
    for i in range(len(all_paths)):
        path = all_paths[i]
        if len(path.split('.')) > 1:
            continue
        print('processing', os.path.join(original_label_dir, path))
        print(count + 1, 'file processed')
        process_a_frame(os.path.join(original_label_dir, path), count, bin_save_dir, label_save_dir)
        count += 1
