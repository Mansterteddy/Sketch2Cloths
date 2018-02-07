from PIL import Image
import os
import numpy as np
import cv2

fold_A = "./compose/"
fold_B = "./groundTruth/"
fold_AB = "./pair/"

fold_AB_train = fold_AB + "train/"
fold_AB_val = fold_AB + "val/"
fold_AB_test = fold_AB + "test/"

os.makedirs(fold_AB_train)
os.makedirs(fold_AB_val)
os.makedirs(fold_AB_test)

splits = os.listdir(fold_A)
count = 0

for sp in splits:

    count += 1

    img_A = os.path.join(fold_A, sp)
    img_B = os.path.join(fold_B, sp)

    im_A = cv2.imread(img_A, cv2.IMREAD_COLOR)
    im_A = cv2.resize(im_A, (550, 800))
    im_B = cv2.imread(img_B, cv2.IMREAD_COLOR)
    im_B = cv2.resize(im_B, (550, 800))

    im_AB = np.concatenate([im_A, im_B], 1)

    if count >= 1 and count <= 900:
        cv2.imwrite(fold_AB_train + sp, im_AB)
    elif count > 900 and count <= 950:
        cv2.imwrite(fold_AB_test + sp, im_AB)
    else:
        cv2.imwrite(fold_AB_val + sp, im_AB)