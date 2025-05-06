from Instance_Segmentation.train_model import get_model_instance_segmentation, get_transform

import torch
import os
from PIL import Image, ImageFilter
import numpy as np
from torchvision.io import read_image
from skimage import morphology
from scipy import ndimage as ndi
import cv2

def erode(cycles, image):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MinFilter(3))
    return image


def dilate(cycles, image):
    for _ in range(cycles):
         image = image.filter(ImageFilter.MaxFilter(3))
    return image
    
def smooth_edges(img):
    img_gray = img.convert("L")
    threshold = 0
    img_threshold = img_gray.point(
        lambda x: 255 if x > threshold else 0
    )
    step_1 = erode(3, img_threshold)
    mask = erode(1, step_1)
    mask = mask.convert("L")
    mask = mask.filter(ImageFilter.BoxBlur(3))
    blank = img.point(lambda _: 0)
    segmented = Image.composite(img, blank, mask)
    return segmented

def smooth_mask(img):
    img_gray = img.convert("L")
    threshold = 0
    img_threshold = img_gray.point(
        lambda x: 255 if x > threshold else 0
    )
    step_1 = erode(3, img_threshold)
    mask = erode(1, step_1)
    mask = mask.convert("L")
    mask = mask.filter(ImageFilter.BoxBlur(3))
    mask = mask.point(
        lambda x: 255 if x > threshold else 0
    )
    mask = dilate(3, mask)
    return mask

class SegmentationModelWrapper():
    def __init__(self, segmentation_model_path):
        self.device = torch.device("cuda")

        self.segmentation_model = get_model_instance_segmentation(2, False)
        self.segmentation_model.load_state_dict(torch.load(
            segmentation_model_path, map_location=self.device))
        self.segmentation_model.to(self.device)
        self.segmentation_model.eval()

        self.score_threshold = 0.9
        self.mask_threshold = 0.9
        self.padding = 10

    def __call__(self, image_path, output_dir):
        name = os.path.basename(image_path)

        image_orig = Image.open(image_path).convert("RGB")
        image_orig = np.array(image_orig)
        image = read_image(image_path).float() / 255.0
        # print(image.max())

        eval_transform = get_transform(train=False)
        with torch.no_grad():
            x = image  # eval_transform(image)
            # convert RGBA -> RGB and move to device
            x = x[:3, ...].to(self.device)
            self.segmentation_model.eval()
            predictions = self.segmentation_model([x,])
            pred = predictions[0]

        masks: np.ndarray = pred["masks"].cpu().numpy()
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        med_mas_size = np.median(masks.sum(axis=(1, 2, 3)))

        # image = image.cpu().numpy()
        # image_orig = np.transpose(image_orig, (1, 2, 0))

        extracted_objs = []

        for i in range(len(masks)):
            if scores[i] >= self.score_threshold:
                # check if mask has intersection with others:
                is_duplicate = False
                for j in range(len(masks)):  # TODO: check for optimization
                    if scores[j] >= self.score_threshold and i != j:
                        intersection = np.count_nonzero(
                            np.logical_and(masks[i, 0] > self.mask_threshold,  masks[j, 0] > self.mask_threshold))
                        mask2_size = np.count_nonzero(masks[j, 0] > self.mask_threshold)
                        # print(f"Intersection size: {intersection}, threshold: {0.3 * mask2_size}")
                        if intersection > 0.7 * mask2_size: # This mask overlaps more than 70% of another mask, thus it is duplicate
                            is_duplicate = True
                            print(f"{i} is duplicate of {j}")
                            break
                if is_duplicate:
                    continue
                mask = masks[i, 0]
                mask = (mask > self.mask_threshold).astype(np.uint8)
                mask = ndi.binary_fill_holes(mask)
                mask = morphology.binary_erosion(mask)
                mask = morphology.binary_dilation(mask)
                mask = morphology.binary_dilation(mask).astype(np.uint8)
                mask = np.array(smooth_mask(Image.fromarray(mask)))

                box = boxes[i]

                extracted_obj = cv2.bitwise_and(
                    image_orig, image_orig, mask=mask
                )
                minr, minc, maxr, maxc = box
                minr, minc, maxr, maxc = max(0, minr-self.padding), max(0, minc-self.padding), min(
                    image_orig.shape[1]-1, maxr+self.padding), min(image_orig.shape[0]-1, maxc+self.padding)

                extracted_obj = extracted_obj[
                    int(minc): int(maxc), int(minr): int(maxr)
                ]
                # extracted_obj = clean_background(extracted_obj)

                output_path = os.path.join(output_dir,
                                           os.path.basename(
                                               os.path.dirname(image_path)),
                                           f'{name.split(".")[0]}_{i}.png')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, extracted_obj[:, :, ::-1])
                extracted_objs.append(Image.fromarray(extracted_obj) )
        return extracted_objs
