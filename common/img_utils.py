import cv2
import numpy as np
from PIL import Image, ImageOps

def clean_background(src:np.ndarray) -> np.ndarray:
    """Util method for beatufication of images. 
    Replace black background with white one.

    ATTENTION: DO NOT USE IT IN DATA PROCESSING PIPELINE
    USE ONLY FOR VISUALIZATION

    Args:
        src (np.ndarray): RGB image to remove black background

    Returns:
        np.ndarray: new image with whitened BG
    """

    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    return dst

def pad2size(image, size=(448, 448)):
    return ImageOps.pad(image, size)

def scale2factor(image, factor=0.5):
    return ImageOps.scale(image, factor)