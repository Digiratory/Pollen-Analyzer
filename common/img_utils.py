import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage import io
from skimage.color import label2rgb, rgb2gray
from skimage import morphology
from skimage.morphology import closing, square
from skimage.filters import threshold_yen
from skimage.segmentation import clear_border, watershed
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

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
    return ImageOps.pad(image, size, color="#000")

def scale2factor(image, factor=0.75):
    return ImageOps.scale(image, factor)


def extract_objects(image_rgb, labels_image, min_distance, padding):
    labels = ndi.binary_fill_holes(labels_image)
    labels = morphology.binary_erosion(labels)
    labels = morphology.binary_dilation(labels).astype(np.uint8)
    
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(labels)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=labels, min_distance=min_distance)
    if len(coords) > 0:
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=labels)


    masked_rgb_images = []
    for region in regionprops(labels):
        minr, minc, maxr, maxc = region.bbox    
        minr, minc, maxr, maxc = max(0, minr-padding), max(0, minc-padding), min(image_rgb.shape[0]-1, maxr+padding), min(image_rgb.shape[1]-1, maxc+padding)
        label_patch = np.copy(labels[minr:maxr, minc:maxc])
        label_patch = clear_border(label_patch)
        image_rgb_patch = np.copy(image_rgb[minr:maxr, minc:maxc])
        image_rgb_patch = np.where((label_patch > 0)[...,None], image_rgb_patch, 0).astype(np.uint8)
        masked_rgb_images.append(image_rgb_patch)
    return masked_rgb_images
    

def object_separation_stage(image_rgb, labels, area_threshold = 20, padding=10, min_distance=40):
    image_rgb = np.copy(image_rgb)
    masked_rgb_images = []
    for region in regionprops(labels):
        if region.area < area_threshold:
            continue  # Skipping small objects
        minr, minc, maxr, maxc = region.bbox            
        minr, minc, maxr, maxc = max(0, minr-padding), max(0, minc-padding), min(image_rgb.shape[0]-1, maxr+padding), min(image_rgb.shape[1]-1, maxc+padding)
        
        label_patch = np.copy(labels[minr:maxr, minc:maxc])
        label_patch = clear_border(label_patch)
        image_rgb_patch = np.copy(image_rgb[minr:maxr, minc:maxc])
        image_rgb_patch = np.where((label_patch > 0)[...,None], image_rgb_patch, 0).astype(np.uint8)
        masked_rgb_images.extend(extract_objects(image_rgb_patch, label_patch, min_distance, padding))
    return masked_rgb_images

def object_detection_stage(image_rgb, radius=25):
    image_rgb = np.copy(image_rgb)
    image = rgb2gray(image_rgb)
    # apply threshold
    thresh = threshold_yen(image)
    bw = closing(image < thresh, square(25))
    bw =  morphology.remove_small_holes(
        morphology.remove_small_objects(bw, 5000), 5000
    )
    bw = morphology.isotropic_erosion(bw, radius)
    bw = morphology.isotropic_dilation(bw, radius)
    label_image = label(bw)
    return label_image
