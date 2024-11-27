import os
import torch
import numpy as np

from torchvision.io import read_image
from torchvision.io import read_image
from Instance_Segmentation import utils
from torchvision.ops.boxes import masks_to_boxes
from Instance_Segmentation.engine import evaluate
from Instance_Segmentation.image_from_mask import cvatParse
from Instance_Segmentation.train_model import get_model_instance_segmentation


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'images/Test/Sample_1'))))
        self.namespace = cvatParse(path_to_xml=os.path.join(root, 'annotations.xml'))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images/Test/Sample_1', self.imgs[idx])
        img = read_image(img_path).float() / 255.0
        
        img_name = self.imgs[idx].split('.')[0]
        masks = [item['mask'] for item in self.namespace[img_name]]
        
        if len(masks) == 0:
            masks = torch.zeros((0, img.shape[1], img.shape[2]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            num_objs = 0
        else:
            masks = torch.tensor(np.array(masks), dtype=torch.uint8)
            num_objs = len(masks)
            boxes = masks_to_boxes(masks)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "masks": masks,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    test_root = r'Data/CVAT_dataset'
    model_root = r'Save/models/InSegModel_fb'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = get_model_instance_segmentation(num_classes=2, pre_trained=False)
    model.load_state_dict(torch.load(model_root, map_location=device))
    model.to(device)
    model.eval()
    dataset_test = SampleDataset(root=test_root)

    data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=6,
    shuffle=False,
    collate_fn=utils.collate_fn,
    )    

    # Output mask & boxes on a given image
    # get_image(model, device, read_image(os.path.join(test_root, 'images/Test/Sample_1/Sample_1 (164).jpg')), 1, 'Save')

    evaluate(model, data_loader_test, device)
