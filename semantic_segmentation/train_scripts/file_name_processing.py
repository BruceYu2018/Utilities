import os
import cv2

file_path = "/cv/DataSet/craneImageData/LabeledData/beibuwan_side_result"

file_names = os.listdir(file_path)
file_names.sort()

count = 96
for file_name in file_names:
    if os.path.splitext(file_name)[-1] == ".jpg":
        file_name_new = str(count) + ".png"
        image = cv2.imread(os.path.join(file_path, file_name))
        cv2.imwrite(os.path.join(file_path, file_name_new), image)
        os.remove(os.path.join(file_path, file_name))
        print file_name_new
    if os.path.splitext(file_name)[-1] == ".json":
        file_name_new = str(count) + "_polygons.json"
        os.rename(os.path.join(file_path, file_name), os.path.join(file_path, file_name_new))
        print file_name_new
        count = count + 1
