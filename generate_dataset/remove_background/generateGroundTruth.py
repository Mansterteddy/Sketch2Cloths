import cv2
import numpy as np
import scipy.io
import os

cloth_image_path = "./dataset/photos/"
label_annotations_path = "./annotations/pixel-level/"
segImage_path = "./groundTruth/"

label_dirs = os.listdir(label_annotations_path)

reserved_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                  53, 54, 55, 56, 57, 58]
removed_class = [0]

for file in label_dirs:
    if file == ".DS_Store": continue

    file = file[:-4]

    image_filename = cloth_image_path + file + ".jpg"
    label_filename = label_annotations_path + file + ".mat"

    label_data = scipy.io.loadmat(label_filename)
    label_data = label_data['groundtruth']

    x_list = np.array([])
    y_list = np.array([])

    for item in removed_class:
        y, x = np.where(label_data == item)
        x_list = np.append(x_list, x)
        y_list = np.append(y_list, y)

    image_data = cv2.imread(image_filename)

    if len(x_list) == 0:
        continue

    else:
        
        for i in range(len(x_list)):
            x_pos = int(x_list[i])
            y_pos = int(y_list[i])
            image_data[y_pos][x_pos] = np.array([255, 255, 255], dtype="uint8")

        cv2.imwrite(segImage_path + file + ".jpg", image_data)