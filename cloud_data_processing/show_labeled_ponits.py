import os

import numpy as np
import open3d as o3d
import sys
sys.path.append('/home/westwell/Desktop/show_cloudponit')
# from xyz2kitti_new_2 import *
def custom_draw_geometry(pcd,linesets):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(linesets)
    render_option = vis.get_render_option()
    render_option.point_size = 4
    render_option.background_color = np.asarray([0, 0, 0])
    vis.run()
    vis.destroy_window()


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


def get_ponits(path):
    f = open(path, 'r')
    all_lines = f.readlines()
    op3d_info = np.zeros((len(all_lines), 3), np.float32)

    info = all_lines[0].split(" ")
    if len(info) == 5:
        label = all_lines[0].split(" ")[-1].split(".")[0]
    elif len(info) == 9:
        label = all_lines[0].split(" ")[5].split(".")[0]
        print(all_lines[0].split(" "),"label")
    else:
        label = None

    cnt = 0
    for one_line in all_lines:
        info = one_line.split(" ")
        op3d_info[cnt][0:3] = info[0:3]
        cnt += 1
    f.close()
    return op3d_info, label


if __name__ == '__main__':
    root = '/cv/DataSet/LidarData/RawData/point/pcd_datasets_result/lcb_pcd_result'
    dirs = []
    for parent, dir_names, file_names in os.walk(root):
        for dir_name in sorted(dir_names):
            dirs.append(os.path.join(parent, dir_name))
    # dir_name
    dirs = sorted(dirs)


    for dir in dirs:
        asc_files = os.listdir(dir)
        all_xyz = []
        cnt = 0
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # render_option = vis.get_render_option()
        # render_option.point_size = 4
        # render_option.background_color = np.asarray([0, 0, 0])
        print("点云路径", dir)
        for acs_file in asc_files:
            point_clouds, label = get_ponits(dir + '/' + acs_file)

            x, y, z, w, h, l = get_xyz_lwh(point_clouds) #中心点,长宽高
            points_box = get_points_box(x, y, z, l, w, h) #box8个坐标点

            lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                                  [0, 4], [1, 5], [2, 6], [3, 7]])
            colors = np.array([[0, 1, 0] for j in range(len(lines_box))])
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_box)
            line_set.lines = o3d.utility.Vector2iVector(lines_box)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(point_clouds[:, :3])
            color = []
            if label == '1':
                color = [255, 0, 0]
            elif label == '2':
                color = [0, 255, 0]
            elif label == '3':
                color = [0, 0, 255]
            elif label == '4':
                color = [255, 255, 0]
            else:
                color = [0, 0, 0]
            point_cloud.paint_uniform_color(color)
            vis.add_geometry(point_cloud)
            vis.add_geometry(line_set)
        # FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
        # vis.add_geometry(FOR1)
        vis.run()
        vis.destroy_window()

