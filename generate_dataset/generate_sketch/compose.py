import cv2
import numpy as np
import scipy.io
import os

dataset_dir = "/Users/manster/Documents/Dataset/"
clothEdge_image_path = dataset_dir + "sketch2img/segClothSketch/"
skin_path = dataset_dir + "sketch2img/skin/"
compose_path = dataset_dir + "sketch2img/compose/"

cur_dirs = os.listdir(clothEdge_image_path)

res_dirs = os.listdir(compose_path)

for file in cur_dirs:
    if file == ".DS_Store": continue

    if file in res_dirs: continue

    file = file[:-4]

    filename_1 = clothEdge_image_path + file + ".jpg"
    filename_2 = skin_path + file + ".jpg"

    data_1 = cv2.imread(filename_1, 1)
    data_2 = cv2.imread(filename_2, 1)

    for i in range(len(data_1)):
        for j in range(len(data_1[0])):
            #print(data_2[i][j])
            if list(data_2[i][j]) != [255, 255, 255]:
                data_1[i][j] = data_2[i][j]

    cv2.imwrite(compose_path + file + ".jpg", data_1)