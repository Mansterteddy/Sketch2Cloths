import cv2
import numpy as np
import os

dataset_dir = "/Users/manster/Documents/Dataset/"
model_path = "/Users/manster/Documents/SourceTree/Sketch2Cloths/Image2Sketch/StructuredEdgeDetection_py/model.yml.gz" # structure forest model
cloth_image_path = dataset_dir + "sketch2img/segCloth/"
segImage_path = dataset_dir + "sketch2img/segClothSketch/"

image_dirs = os.listdir(cloth_image_path)

for file in image_dirs:

    if file == ".DS_Store": continue

    file = file[:-4]

    print("cur file: ", file)

    image_filename = cloth_image_path + file +'.jpg'
    image_data = cv2.imread(image_filename, 1)

    img_float = np.float32(image_data)
    img_float = img_float*(1.0/255.0)
    model = cv2.ximgproc.createStructuredEdgeDetection(model_path)
    img_edge = 1 - model.detectEdges(img_float)
    cv2.imwrite(segImage_path + file + '.jpg', img_edge * 255.0)