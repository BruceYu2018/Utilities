import os
import math
import numpy as np

degree = math.pi / 2
rotateMatrix = np.array([[round(math.cos(degree), 4), 0, round(-math.sin(degree), 4)],
                         [0, 1, 0],
                         [round(math.sin(degree), 4), 0, round(math.cos(degree), 4)]])


def readLabel(filepath):
    label = []
    with open(filepath, 'r') as f:
        line = f.readline().strip()
        while line:
            linestr = line.split("\t")
            linestr[2] = str(round(float(linestr[2]), 2))
            linestr[3] = str(round(float(linestr[3]), 2))
            linestr[4] = str(round(float(linestr[4]), 2))
            linestr[5] = str(round(float(linestr[5]), 2))
            linestr[6] = str(round(float(linestr[6]), 2))
            linestr[7] = str(round(float(linestr[7]), 2))
            label.append(linestr)
            line = f.readline().strip()
    return label


def convert(oldLabel, newLabel):
    ori_path = oldLabel
    file_list = os.listdir(ori_path)
    file_list.sort()
    des_path = newLabel
    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)
    count = 0
    for file in file_list:
        (filename, extension) = os.path.splitext(file)
        old_label_file = os.path.join(ori_path, filename) + '.txt'
        print old_label_file
        labels = readLabel(old_label_file)
        new_label_file = os.path.join(newLabel, str(count).zfill(6)) + '.txt'
        f = open(new_label_file, 'w')
        new_label_lists = []
        for label in labels:
            # pos = np.array([[label[5], label[6], label[7]]], dtype=np.float32)
            # pos = np.dot(pos, rotateMatrix)
            # pos[0, 2] = pos[0, 2] + 30
            # pos[0, 0] = pos[0, 0] + 40
            # new_label_list = [label[0], '0.00', label[1], '-10', '-1', '-1', '-1', '-1',
            #                   label[4], label[3], label[2],
            #                   str(round(pos[0, 0], 2)), str(round(pos[0, 1], 2)), str(round(pos[0, 2], 2)), label[9]]
            new_label_list = [label[0], '0.00', label[1], '-10', '-1', '-1', '-1', '-1',
                              label[4], label[3], label[2],
                              label[5], label[6], label[7], label[9]]
            print new_label_list
            new_label_lists.append(' '.join(new_label_list))
        f.writelines(new_label_lists)
        count += 1


if __name__ == "__main__":
    oldLabel = "/cv/DataSet/LidarData/RawData/lidar_annotation/_bbox"
    newLabel = "/cv/DataSet/LidarData/TrainData/training/label_2"
    convert(oldLabel, newLabel)