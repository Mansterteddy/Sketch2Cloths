import cv2
import numpy as np
import os

model_path = "/Users/manster/Documents/SourceTree/Sketch2Cloths/Image2Sketch/StructuredEdgeDetection_py/model.yml.gz" # structure forest model
cloth_image_path = "/Users/manster/Documents/SourceTree/Sketch2Cloths/Image2Sketch/StructuredEdgeDetection_py/img/"
segImage_path = "/Users/manster/Documents/SourceTree/Sketch2Cloths/Image2Sketch/StructuredEdgeDetection_py/res/"

image_dirs = os.listdir(cloth_image_path)

for file in image_dirs:

    file = file[:-4]

    image_filename = cloth_image_path + file +'.jpg'
    image_data = cv2.imread(image_filename, 1)

    img_float = np.float32(image_data)
    img_float = img_float*(1.0/255.0)
    model = cv2.ximgproc.createStructuredEdgeDetection(model_path)
    img_edge = 1 - model.detectEdges(img_float)
    cv2.imwrite(segImage_path + file + '.jpg', img_edge * 255.0)