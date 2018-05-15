import cv2
import numpy as np
import os

model_path = "E:/workSpace/AI/edgeDetection/resources/model.yml.gz" # structure forest model
cloth_image_path = "./segCloth_version2/"
segImage_path = "./segClothEdge_version2/"

image_dirs = os.listdir(cloth_image_path)

for file in image_dirs:
    file = file[:-4]

    image_filename = cloth_image_path + file +'.jpg'
    image_data = cv2.imread(image_filename,1)
    #cv2.imshow('res',image_data)
    #cv2.waitKey(0)
    if image_data.size == 0:
        print('cannot read file')
        exit(0)
    img_float = np.float32(image_data)
    img_float = img_float*(1.0/255.0)
    retval = cv2.ximgproc.createStructuredEdgeDetection(model_path)
    img_edge = 1 - retval.detectEdges(img_float)
    cv2.imwrite(segImage_path + file+'.jpg',img_edge*255.0)
    #cv2.imshow('result',img_edge)
    print(file)
    #cv2.waitKey(0)
