import cv2
import numpy as np
import scipy.io
import os

dataset_dir = "/Users/manster/Documents/Dataset/"
cloth_image_path = dataset_dir + "sketch2img/groundTruth/"
label_annotations_path = dataset_dir + "co-parsing-raw/annotations/pixel-level/"
segImage_path = dataset_dir + "sketch2img/skin/"

label_dirs = os.listdir(label_annotations_path)

all_class = [i for i in range(59)]
removed_class = [0, 4, 5, 6, 10, 11, 13, 14, 22, 24, 25, 26, 27, 30, 31, 35, 38, 40,
                    42, 46, 48, 49, 50, 51, 53, 54, 55]

for file in label_dirs:
    if file == ".DS_Store": continue

    print("cur file: ", file)

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