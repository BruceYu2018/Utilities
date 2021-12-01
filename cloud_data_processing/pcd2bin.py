import os
import numpy as np
import math

degree = math.pi / 2
rotateMatrix = np.array([[round(math.cos(degree), 4), 0, round(-math.sin(degree), 4)],
                         [0, 1, 0],
                         [round(math.sin(degree), 4), 0, round(math.cos(degree), 4)]])


def read_pcd(filepath):
    lidar = []
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split(" ")
            if len(linestr) == 4:
                linestr_convert = list(map(float, linestr))
                lidar.append(linestr_convert)
            line = f.readline().strip()
    return np.array(lidar)


def convert(pcdfolder, binfolder):
    ori_path = pcdfolder
    file_list = os.listdir(ori_path)
    file_list.sort()
    des_path = binfolder
    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)
    count = 0
    for file in file_list:
        print file
        (filename, extension) = os.path.splitext(file)
        velodyne_file = os.path.join(ori_path, filename) + '.pcd'
        pl = read_pcd(velodyne_file)
        pl = pl.reshape(-1, 4).astype(np.float32)
        # print pl
        # points = pl[:, :3]
        # points = np.dot(points, rotateMatrix)
        # pl[:, :3] = points
        # pl[:, 2] = pl[:, 2] + 30
        # pl[:, 0] = pl[:, 0] + 40
        # pl = pl.reshape(-1, 4).astype(np.float32)
        # print pl
        velodyne_file_new = os.path.join(des_path, str(count).zfill(6)) + '.bin'
        print velodyne_file_new
        pl.tofile(velodyne_file_new)
        count += 1


if __name__ == "__main__":
    pcdfolder = "/cv/DataSet/LidarData/RawData/horizon_pcd"
    binfolder = "/cv/Git_Projects/mmdetection3d/data/horizon/training/velodyne"
    convert(pcdfolder, binfolder)