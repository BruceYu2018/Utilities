# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:59:36 2019

@author: shen1994
"""
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import random

if __name__ == "__main__":
    dataset_dir = '/cv/DataSet/craneImageData/TrainData/CraneDoor/'
    png_list = os.listdir(dataset_dir + 'png')
    png_list.sort()
 
    # split data for training
    train_f = open(dataset_dir + 'train_list.txt', 'w+')
    test_f = open(dataset_dir + 'test_list.txt', 'w+')

    interval = 10
    start = 0
    end = start + interval
    skipList = []
    while end < len(png_list):
        skipID = random.randint(start, end)
        skipList.append(skipID)
        start = end + 1
        end = start + interval
    print skipList
    print len(skipList)

    for index, onedir in enumerate(png_list):
        rgb_dir = dataset_dir + 'png/' + onedir
        depth_dir = dataset_dir + 'Tlabel/' + onedir
        label_dir = dataset_dir + 'Tlabel/' + onedir
        if index in skipList:
            test_f.write(rgb_dir + ' ' + depth_dir + ' ' + label_dir + '\n')
        else:
            # train_f.write(rgb_dir + ' ' + depth_dir + ' ' + label_dir + '\n')
            train_f.write(rgb_dir + ' ' + depth_dir + ' ' + label_dir + '\n')

    train_f.close()
    test_f.close()
