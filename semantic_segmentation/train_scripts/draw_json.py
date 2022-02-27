import cv2
import numpy as np
import json
import os
from PIL import Image

color_dict = {'top': (128, 128, 128), 'backdoor': (0, 255, 0), 'frontdoor': (255, 128, 255), 'pole': (0, 255, 255),
              'lockhole': (0, 0, 255), 'black': (255, 255, 0), 'background': (128, 255, 0), 'seal': (0, 255, 128),
              'tray_head': (255, 128, 0), 'handle': (128, 128, 0), 'side': (128, 255, 255), 'lift': (255, 0, 255),
              'liewen': (255, 128, 255), 'podong': (255, 128, 255), 'sidewalk': (128, 128, 0), "back": (0, 255, 0),
              'front': (255, 128, 255)}

label_dict = {'top': 1, 'backdoor': 2, 'frontdoor': 3, 'pole': 4, 'lockhole': 5, 'black': 6, 'background': 7, 'seal': 8,
              'tray_head': 9, 'handle': 10, 'side': 11, 'lift': 12, 'liewen': 3, 'podong': 3, 'sidewalk': 10,
              'back': 2, 'front': 3}

image_path = "/cv/DataSet/craneImageData/TrainData/1.DataGenerate/png/"
json_path = "/cv/DataSet/craneImageData/TrainData/1.DataGenerate/json/"
save_path = "/cv/DataSet/craneImageData/TrainData/1.DataGenerate/label/"
save2_path = "/cv/DataSet/craneImageData/TrainData/1.DataGenerate/image2label/"
save3_path = "/cv/DataSet/craneImageData/TrainData/1.DataGenerate/Tlabel/"
image_names = os.listdir(image_path)
print len(image_names)
image_names.sort()

is_break = False
file_dict_items = [('container_truck_background_leiwen_hole',
                    ['pole', 'lockhole', 'black', 'seal', 'handle', 'backdoor', 'top', 'lift', 'side', 'tray_head',
                     'background', 'liewen', 'podong', 'frontdoor', 'sidewalk', 'back', 'front'])]
for image_name in image_names:

    image = cv2.imread(image_path + image_name)
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    smask = np.full((image.shape[0], image.shape[1]), 0, dtype=np.uint8)
    test = image_name.split(".")[0]
    print(test)
    for fk, fv in file_dict_items:
        buff = json_path + test + '_polygons.json'
        if not os.path.exists(buff):
            continue

        with open(buff, "r") as f:
            jsonstr = f.read()
        mjson = json.loads(jsonstr)
        p_objects = mjson['objects']

        for object_index in range(len(p_objects)):
            deleted = p_objects[object_index]['deleted']
            if deleted == 1:
                continue
            label = p_objects[object_index]['label']

            color = (0, 0, 0)
            slabel = 0
            for v in fv:
                if label == v:
                    color = color_dict[v]
                    slabel = label_dict[v]

            polygon = mjson['objects'][object_index]['polygon']
            polygon_mod = np.asarray(polygon).astype('int32')
            polygon_mod.resize((1, len(polygon), 2))
            cv2.polylines(image, polygon_mod, 1, color)
            cv2.fillPoly(mask, polygon_mod.copy(), color)
            cv2.fillPoly(smask, polygon_mod.copy(), slabel)

    cv2.imwrite(save_path + test + ".png", mask)
    cv2.imwrite(save2_path + test + ".png", np.vstack((image, mask)))

    # img = Image.fromarray(smask.astype('uint8')).convert('L')
    # img = img.resize((900,500), Image.ANTIALIAS)

    cv2.imwrite(save3_path + test + ".png", smask)
    '''
    while(True):

        cv2.imshow("image", np.hstack((image, mask)))

        if cv2.waitKey(5) & 0xFF == ord(' '):
            break
        if cv2.waitKey(5) & 0xFF == ord('q'):
            is_break = True
            break

    if is_break:
        break
    '''
