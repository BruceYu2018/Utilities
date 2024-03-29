import os
import numpy as np
import struct
import open3d


def read_bin_velodyne(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def main():
    root_dir = '/cv/DataSet/LidarData/TrainData/training/velodyne/'
    # root_dir = '/cv/Git_Projects/mmdetection3d/data/kitti/training/velodyne'
    filename = os.listdir(root_dir)
    file_number = len(filename)
    FOR1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
    pcd = open3d.open3d.geometry.PointCloud()

    for i in range(file_number):
        path = os.path.join(root_dir, filename[i])
        print(path)
        example = read_bin_velodyne(path)
        # From numpy to Open3D
        pcd.points = open3d.open3d.utility.Vector3dVector(example)
        open3d.open3d.visualization.draw_geometries([FOR1, pcd])


if __name__ == "__main__":
    main()
