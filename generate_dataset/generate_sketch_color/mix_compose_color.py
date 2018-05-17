import numpy as np
import cv2
import random
import scipy.io
import os

dataset_dir = "/Users/manster/Documents/Dataset/"
orginal_image_path = dataset_dir + "sketch2img/groundTruth/"
compose_image_path = dataset_dir + "sketch2img/compose/"
label_annotations_path = dataset_dir + "co-parsing-raw/annotations/pixel-level/"
mix_image_path = dataset_dir + 'sketch2img/compose_color/'


label_dirs = os.listdir(label_annotations_path) # 读取mat的路径

"""label_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27,
                  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                  53, 54, 55, 56, 57, 58]
    label_class = [2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27,
                 28, 30, 31, 33, 35, 36, 37, 38, 39, 40, 42, 43, 45, 46, 48, 49, 50, 51,
                 53, 54, 55, 58]"""
label_class = [4, 5, 6, 10, 11, 13, 14, 22, 24, 25, 26, 27, 30, 31, 35, 38, 40,
                    42, 46, 48, 49, 50, 51, 53, 54, 55]  

for file in label_dirs:

    file = file[:-4]                            
    
    compose_image_filename = compose_image_path + file + '.jpg'
    image_filename = orginal_image_path + file +'.jpg'
    label_filename = label_annotations_path + file + '.mat'

    label_data = scipy.io.loadmat(label_filename)
    label_data = label_data['groundtruth']    

    x_list = np.array([])
    y_list = np.array([])
    
    item_dict = dict()                         # 存储类别和下标的关系
    sample_dict = dict()                       # 存储所有取样的图片和下标，key = 下标

    kernel_size = 5                            # 采样块大小
    valid_flag = 1                             # 是否采样成功
    for i_item in label_class:
        x, y = np.where(label_data == i_item)     # 将label_data中item的下标存到(y,x)
        if len(x) == 0:
            continue
        else:
            item_dict[i_item] = np.array([x,y])

    compose_image_data = cv2.imread(compose_image_filename)    # 读取compose图片
    image_data = cv2.imread(image_filename)                    # 读取orginal图片
    if len(item_dict) == 0:                                    # 图中一个类别都没有
        continue
    else:
        for item in item_dict:                                 # 进入到一类物品里头
            if file == '0003' and item == 24: #过滤0003图片中的jacket
                continue
            sample_num = 10                                    # 采样个数
            while sample_num:
                x_random = random.randint(0, compose_image_data.shape[0])# 随机产生坐标
                y_random = random.randint(0, compose_image_data.shape[1])
                for stribe in range(kernel_size):
                    x_flag = np.where(item_dict[item][0] == x_random+stribe)
                    y_flag = np.where(item_dict[item][1] == y_random+stribe)
                    if len(x_flag[0]) == 0 or len(y_flag[0]) == 0:
                        valid_flag = 0
                        break
                if valid_flag != 0:           # 成功取样
                    region = image_data[x_random:x_random+kernel_size,y_random:y_random+kernel_size]
                    sample_dict[(x_random,y_random)] = region     #保存对应的取样点以及采样图片
                    sample_num -= 1
                    #print('item',item,'sample',sample_num)
                else:
                    valid_flag = 1
    for box, region in sample_dict.items():
        compose_image_data[box[0]:box[0]+kernel_size,box[1]:box[1]+kernel_size] = region
    cv2.imwrite(mix_image_path + file + ".jpg", compose_image_data)
    print(file)

