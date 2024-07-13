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
from torchvision import transforms, datasets, tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from engine import train_one_epoch, evaluate


def get_mask(src):
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp, 0,255, cv2.THRESH_MASK)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    return dst


def clean_background(src):
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    return dst


class AugmentImageDataset(datasets.DatasetFolder):
    def __init__(self, object_dir: str, back_dir: str, package=100, back_usages=2, objects_per_image=(1, 2),
                 transform=None, transform_object=None):
        self.img_dir = object_dir
        self.back_dir = back_dir
        self.objects_per_image = objects_per_image
        self.package = package
        self.back_usages = back_usages
        self.transform = transform
        self.transform_object = transform_object
        self.objects = []
        self.backgrounds = []

        for folder_name in os.listdir(object_dir):
            folder_path = os.path.join(object_dir, folder_name)
            for image_name in os.listdir(folder_path):
                image = Image.open(os.path.join(folder_path, image_name)).convert('RGB')
                self.objects.append([image, folder_name])

        for image_name in os.listdir(back_dir):
            image = F.to_pil_image(read_image(os.path.join(back_dir, image_name)))
            self.backgrounds.append(image)

    def __len__(self):
        return len(self.backgrounds) * self.package * self.back_usages

    def __getitem__(self, idx):
        idx //= self.package * self.back_usages
        close_param = 0.4
        back = deepcopy(self.backgrounds)[idx]
        objects = deepcopy(self.objects)
        objects_per_image = randint(*self.objects_per_image)

        busy_places = []
        target = {}
        masks = []
        itr = 0
        while len(busy_places) < objects_per_image and itr < objects_per_image * 3:
            itr += 1
            object = objects[randrange(len(self.objects))]
            object_image = F.to_pil_image(clean_background(np.array(object[0])))

            if self.transform_object is not None:
                object_image = self.transform_object(object_image)

            mask_background = F.to_pil_image(np.zeros((back.size[1], back.size[0], 1)))
            pos_x = randrange(0, back.size[0] - object_image.size[0])
            pos_y = randrange(0, back.size[1] - object_image.size[1])
            center_dot = ((pos_x + object_image.size[0]) // 2, (pos_y + object_image.size[1]) // 2)
            radius = max(object_image.size) // 2

            for bp in busy_places:
                if ((bp[0][0] - center_dot[0]) ** 2 + (bp[0][1] - center_dot[1]) ** 2) ** 0.5 < (
                        radius + bp[1]) * close_param:
                    break
            else:
                mask_background.paste(object_image, (pos_x, pos_y), object_image)
                mask = np.array(mask_background)
                mask[mask != 0] = 1
                masks += [mask] 

                busy_places.append((center_dot, radius))
                back.paste(object_image, (pos_x, pos_y), object_image)

        img = tv_tensors.Image(back)
        masks = torch.from_numpy(np.array(masks)).to(dtype=torch.uint8)
        labels = torch.ones((len(busy_places),), dtype=torch.int64)
        ispollen = torch.zeros((len(busy_places),), dtype=torch.int64)
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
        

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
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
        transforms.append(T.RandomPhotometricDistort(p=1))
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        transforms.append(T.RandomVerticalFlip(p=0.5))
    transforms.append(T.ToDtype(torch.float32, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def get_transform_object():
    transforms = []
    transforms.append(T.RandomHorizontalFlip(p=0.5))
    transforms.append(T.RandomVerticalFlip(p=0.5))
    return T.Compose(transforms)


def divide_dataset(div_point, dataset):
    train_size = int(div_point * len(dataset))
    test_size = len(dataset) - train_size
    return torch.utils.data.random_split(dataset, [train_size, test_size])


def get_image(model, image, epoch, save_dir):
    eval_transform = get_transform(train=False)
    
    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        # convert RGBA -> RGB and move to device
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


def train_model(model, model_save_dir:str, image_save_dir:str, optimizer, data_loader:torch.utils.data.DataLoader, data_loader_test:torch.utils.data.DataLoader,
                device, num_epochs:int, log_dir:str, log_freq: int):
        writer = SummaryWriter(log_dir)

        for epoch in range(1, num_epochs+1):
            # train for one epoch, logging every 10 iterations
            train_one_epoch(model, optimizer, data_loader, writer, device, epoch, print_freq=log_freq)

            # update the learning rate
            lr_scheduler.step()
            
            evaluator = evaluate(model, data_loader_test, writer, device, epoch)
            eval_res = {}
            for iou_type, coco in evaluator.coco_eval.items():
                for itr in range(1, 12 + 1):
                    if itr <= 6: typo = 'AP'
                    else: typo = 'AR'
                    
                    area = 'all'
                    id = itr % 6
                    if id == 0:area = 'lrg'
                    elif id == 1: area = 'med'
                    elif id == 2: area = 'sml'

                    iou = '0.50:0.95'
                    if itr == 2: iou = '0.50'
                    elif itr == 3: iou = '0.75'

                    max_dets = '100'
                    if itr == 7: max_dets = '1'
                    elif itr == 8: max_dets = '10'

                    sum_res  = coco.stats[itr-1]
                    
                    eval_res.update({f'|{iou_type}-{typo}|area:{area}|IoU:{iou}|dets:{max_dets}|': sum_res})
                    
            writer.add_scalars('Validation tables', eval_res, epoch*len(data_loader_test))
            writer.flush()

            if (epoch) % 5 == 0:
                image = f_d[0][0]
                get_image(model, image, epoch, image_save_dir)

        writer.close()
        torch.save(model.state_dict(), model_save_dir)


if __name__ == '__main__':
    # train/save dirs
    noise_dir = 'Data/Noise_40X'
    pollen_dir = 'Data/pollen_dataset_2024_05_08_objects'
    PATH_TO_MODEL = 'Save/models/final_model_v2'
    log_dir = 'Save/metrics_v2/instance_seg'
    image_save_dir = 'Save/image_v2'

    #pollen_classes = [folder for folder in os.listdir(pollen_dir) if folder[0] != '.']
    num_classes = 2  # pollen as one class + background (noise)
    
    # synthetic dataset definition
    f_d = AugmentImageDataset(pollen_dir, noise_dir, package=5, back_usages=2, objects_per_image=(1, 5), transform=get_transform(True),
                        transform_object=get_transform_object())
    
    # load a coco-model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    backbone.out_channels = 1280

    #anchor settings
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    # if  backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # ``OrderedDict[Tensor]``, and in ``featmap_names`` you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    # mixing everything together inside a FasterRCNN model
    model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )


    dataset, dataset_test = divide_dataset(0.7, f_d)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn,
    )


    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    # model = DDP(model, device_ids=[0, 1])
    model.to(device)


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )

    num_epochs = 30
    log_freq = 10

    train_model(model, PATH_TO_MODEL, image_save_dir, optimizer, data_loader, data_loader_test, device, num_epochs, log_dir,  log_freq)
    