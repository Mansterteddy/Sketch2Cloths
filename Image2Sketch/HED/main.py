import tensorflow as tf
import numpy as np
import cv2

import config
import net

def main():
    hed = net.HED(config)
    hed.predict(config.path_img)

if __name__ == "__main__":
    main()
