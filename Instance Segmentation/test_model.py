import os
import torch
import utils
import numpy as np

from engine import evaluate
from image_from_mask import cvatParse
from torchvision.io import read_image
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from train_model import get_model_instance_segmentation


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


def test_model(model, data_loader_test:torch.utils.data.DataLoader, device):
    # Evaluate the model
    evaluate(model, data_loader_test, device)
    
    # eval_res = {}
    # for iou_type, coco in evaluator.coco_eval.items():
    #     for itr in range(1, 12 + 1):
    #         if itr <= 6: typo = 'AP'
    #         else: typo = 'AR'
            
    #         area = 'all'
    #         id = itr % 6
    #         if id == 0:area = 'lrg'
    #         elif id == 1: area = 'med'
    #         elif id == 2: area = 'sml'

    #         iou = '0.50:0.95'
    #         if itr == 2: iou = '0.50'
    #         elif itr == 3: iou = '0.75'

    #         max_dets = '100'
    #         if itr == 7: max_dets = '1'
    #         elif itr == 8: max_dets = '10'

    #         sum_res  = coco.stats[itr-1]
            
    #         eval_res.update({f'|{iou_type}-{typo}|area:{area}|IoU:{iou}|dets:{max_dets}|': sum_res})
            
    # writer.add_scalars('Validation tables', eval_res, epoch*len(data_loader_test))



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

    test_model(model, data_loader_test, device)
