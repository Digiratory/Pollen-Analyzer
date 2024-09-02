import os
import cv2
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from random import randint, randrange
import utils
import torch
import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from engine import train_one_epoch, evaluate


def clean_background(src):
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    return dst


class AugmentImageDataset(datasets.DatasetFolder):
    def __init__(self, object_dir: str, back_dir: str, back_usages=10, objects_per_image=(1, 2),
                 transform=None, transform_object=None):
        self.img_dir = object_dir
        self.back_dir = back_dir
        self.objects_per_image = objects_per_image
        self.back_usages = back_usages
        self.transform = transform
        self.transform_object = transform_object
        self.objects = []
        self.backgrounds = []

        for folder_name in os.listdir(object_dir):
            folder_path = os.path.join(object_dir, folder_name)
            for image_name in os.listdir(folder_path):
                if image_name.endswith('.png'):
                    image = Image.open(os.path.join(folder_path, image_name)).convert('RGB')
                    self.objects.append(image)

        for image_name in os.listdir(back_dir):
            image = F.to_pil_image(read_image(os.path.join(back_dir, image_name)))
            self.backgrounds.append(image)

    def __len__(self):
        return len(self.backgrounds) * self.back_usages

    def __getitem__(self, idx):
        idx //= self.back_usages
        back = deepcopy(self.backgrounds[idx])
        objects_per_image = randint(*self.objects_per_image)

        target = {}
        masks = []
        itr = 0
        fitted = 0
        while fitted < objects_per_image and itr < objects_per_image * 3:
            itr += 1
            object = deepcopy(self.objects[randrange(len(self.objects))])
            object_image = F.to_pil_image(clean_background(np.array(object)))

            if self.transform_object is not None:
                object_image = self.transform_object(object_image)

            pos_x = randrange(0, back.size[0] - object_image.size[0])
            pos_y = randrange(0, back.size[1] - object_image.size[1])
            
            mask_background = F.to_pil_image(np.zeros((back.size[1], back.size[0], 1)))
            mask_background.paste(object_image, (pos_x, pos_y), object_image)
            mask = np.array(mask_background)
            mask[mask != 0] = 1
            
            obj_size = cv2.countNonZero(mask)
            for fitted_mask in masks:
                fitted_obj_size = cv2.countNonZero(fitted_mask)
                if cv2.countNonZero(cv2.bitwise_and(fitted_mask, mask)) > 0.05 * min(obj_size, fitted_obj_size):
                    break
            else:
                masks += [mask] 
                fitted += 1
                back.paste(object_image, (pos_x, pos_y), object_image)

        img = tv_tensors.Image(back)
        masks = torch.from_numpy(np.array(masks)).to(dtype=torch.uint8)
        labels = torch.ones((fitted,), dtype=torch.int64)
        ispollen = torch.zeros((fitted,), dtype=torch.int64)
        boxes = masks_to_boxes(masks)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target['masks'] = tv_tensors.Mask(masks)
        target['boxes'] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(back))
        target['labels'] = labels
        target['iscrowd'] = ispollen # easier to use iscrowd
        target['area'] = area
        target['image_id'] = idx
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        
        return img, target
        

def get_model_instance_segmentation(num_classes, pre_trained: bool):
    # load an instance segmentation model pre-trained on COCO
    if pre_trained:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='COCO_V1')
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2()

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomPhotometricDistort(p=0.5))
        transforms.append(T.RandomAutocontrast(p=0.5))
        transforms.append(T.RandomAdjustSharpness(p=0.5, sharpness_factor=2))
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.RandomVerticalFlip(p=0.5))
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_transform_object():
    transforms = []
    transforms.append(T.RandomRotation(degrees=180))
    return T.Compose(transforms)


def get_image(model, image, epoch, save_dir):
    eval_transform = get_transform(train=False)
    
    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]
    model.train()
    
    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    orig = deepcopy(image)
    pred_labels = [f"pollen: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")
    
    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    plt.figure(figsize=(12, 12))
    plt.imshow(orig.permute(1, 2, 0))
    plt.savefig(f'{save_dir}/orig[{epoch}].png')
    
    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.savefig(f'{save_dir}/img[{epoch}].png')


def freeze_layers(model, amount):
    freezed_params = []
    for param in model.parameters():
        param.requires_grad = False
        freezed_params.append(param)
        if len(freezed_params) == amount:
            return freezed_params


def unfreeze_layers(model):
    for param in model.parameters():
        param.requires_grad = True


def train_model(model, model_save_dir:str, image_save_dir:str, data_loader:torch.utils.data.DataLoader, 
                review_dataset:datasets.DatasetFolder, device, num_epochs:int, freezed_epochs: int, log_dir:str, log_freq: int):
    writer = SummaryWriter(log_dir)

    if freezed_epochs:
        body_len = 191 # amount of layers not belonging to roi_heads
        freezed_params = freeze_layers(model, body_len)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0001
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20, 
        gamma=0.1
    )

    for epoch in range(1, num_epochs+1):
        if freezed_epochs and (epoch - 1 == freezed_epochs):
            unfreeze_layers(model)
            # creating new group for unfreezed parameters
            optimizer.add_param_group({
                'params': freezed_params,
                'lr': 0.0001,
                'momentum': 0.9,
                'weight_decay': 0.00005
            })

        train_one_epoch(model, optimizer, data_loader, writer, device, epoch, print_freq=log_freq)

        lr_scheduler.step()
        
        writer.flush()

        image = review_dataset[randrange(len(review_dataset))][0]
        get_image(model, image, epoch, image_save_dir)

    writer.close()
    torch.save(model.state_dict(), model_save_dir)


if __name__ == '__main__':
    # train/save dirs
    noise_dir = 'Data/Noise_40X_95'
    pollen_dir = '/home/shared/datasets/pollen_dataset_2024_08_24_objects_clean'
    PATH_TO_MODEL = 'Save/models/InSegModel_fb'
    log_dir = 'Save/metrics/log_fb'
    image_save_dir = 'Save/images/images_fb'

    #pollen_classes = [folder for folder in os.listdir(pollen_dir) if folder[0] != '.']
    num_classes = 2  # pollen as one class + background (noise)
    freezed_epochs = 20 # number of epochs when only roi head will get updates

    # synthetic dataset definition
    dataset = AugmentImageDataset(pollen_dir, noise_dir, back_usages=10, objects_per_image=(1, 6), transform=get_transform(True),
                        transform_object=get_transform_object())

    review_dataset = AugmentImageDataset(pollen_dir, noise_dir, back_usages=1, objects_per_image=(1, 6), transform=get_transform(False),transform_object=get_transform_object())

    # define training data loader
    data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    collate_fn=utils.collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model_instance_segmentation(num_classes, pre_trained=True)
    model.to(device)

    num_epochs = 60
    log_freq = 25

    train_model(model, PATH_TO_MODEL, image_save_dir, data_loader, review_dataset, 
                device, num_epochs, freezed_epochs, log_dir,  log_freq)
      