import os
import random


def convert(oldLabel, newLabel):
    ori_path = oldLabel
    file_list = os.listdir(ori_path)
    file_list.sort()
    des_path = newLabel
    if os.path.exists(des_path):
        pass
    else:
        os.makedirs(des_path)
    write_list = []
    train_list = []
    test_list = []
    train_file = open(os.path.join(newLabel, "train.txt"), 'w')
    test_file = open(os.path.join(newLabel, "test.txt"), 'w')
    trainval_file = open(os.path.join(newLabel, "trainval.txt"), 'w')
    val_file = open(os.path.join(newLabel, "val.txt"), 'w')
    interval = 15
    start = 0
    end = start + interval
    skipList = []
    while end < len(file_list):
        skipID = random.randint(start, end)
        skipList.append(skipID)
        start = end + 1
        end = start + interval
    print skipList
    print len(skipList)

    for index, file in enumerate(file_list):
        (filename, extension) = os.path.splitext(file)
        write_list.append(filename + '\n')
        if index in skipList:
            test_list.append(filename + '\n')
        else:
            train_list.append(filename + '\n')
    train_file.writelines(train_list)
    test_file.writelines(test_list)
    trainval_file.writelines(write_list)
    val_file.writelines(write_list)


if __name__ == "__main__":
    oldLabel = "/cv/DataSet/LidarData/TrainDataAvia/training/velodyne"
    newLabel = "/cv/DataSet/LidarData/TrainDataAvia/ImageSets"
    convert(oldLabel, newLabel)