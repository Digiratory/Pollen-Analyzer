import torch
from torch import nn
from PIL import ImageOps
import torchvision.transforms as transforms
from torchvision.transforms import v2
from common.img_utils import pad2size, scale2factor
from torchvision import models


class OpenSetClassificationModel(nn.Module):
    def __init__(self,
                 n_classes,
                 backbone="resnet18",
                 ):
        super().__init__()
        self.num_classes = n_classes
        self.backbone = backbone
        if backbone == "resnet18":
            self.model = models.resnet18(weights='DEFAULT')
            self.model.fc = nn.Linear(512, n_classes, bias=False)
        elif backbone == "resnet34":
            self.model = models.resnet34(weights='DEFAULT')
            self.model.fc = nn.Linear(512, n_classes, bias=False)
        elif backbone == "resnet50":
            self.model = models.resnet50(weights='DEFAULT')
            self.model.fc = nn.Linear(512, n_classes, bias=False)
        elif backbone == "resnet101":
            self.model = models.resnet101(weights='DEFAULT')
            self.model.fc = nn.Linear(512, n_classes, bias=False)
        elif backbone == "resnet152":
            self.model = models.resnet152(weights='DEFAULT')
            self.model.fc = nn.Linear(512, n_classes, bias=False)
        elif backbone == "efficientnet_v2":
            self.model = models.efficientnet_v2_s(weights='DEFAULT')
            self.model.classifier = nn.Linear(512, n_classes, bias=False)
        elif backbone == "inception_v3":
            self.model = models.inception_v3(
                weights='DEFAULT', aux_logits=False)
            self.model.fc = nn.Linear(512, n_classes, bias=False)
        elif backbone == "vit":
            self.model = models.vit_b_16(weights='DEFAULT')
            self.model.heads.head = nn.Linear(512, n_classes, bias=False)
        elif backbone == "swin_transformer":
            self.model = models.swin_t(weights='DEFAULT')
            self.model.head = nn.Linear(512, n_classes, bias=False)
        else:
            raise ValueError("Unsupported backbone")

        self.anchors = nn.Parameter(torch.zeros(
            self.num_classes, self.num_classes).float(), requires_grad=False)

    # def train(self, mode: bool):
    #     super().train(mode)
    #     self.model.eval()
    #     return self

    def forward(self, x, skip_distance=False):
        outLinear = self.model(x)
        outDistance = None if skip_distance else self.distance_classifier(
            outLinear)

        return outLinear, outDistance

    def distance_classifier(self, x):
        ''' Calculates euclidean distance from x to each class anchor
            Returns n x m array of distance from input of batch_size n to anchors of size m
        '''

        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.linalg.norm(x-anchors, ord=2, dim=2)

        return dists

    def set_anchors(self, means):
        self.anchors = nn.Parameter(means.float(), requires_grad=False)


def distance_to_scores(distances):
    softmax = torch.nn.Softmax(dim=1)
    softmin = softmax(-distances)
    invScores = 1-softmin
    scores = distances*invScores
    return scores


def scores_to_class(scores: torch.Tensor, threshold):
    candidate = scores.argmin(dim=1)
    pred_candidate = torch.where(
        scores[range(scores.shape[0]), candidate.T] < threshold, candidate, -1)
    return pred_candidate


def get_static_transforms():
    return v2.Compose([
        v2.Lambda(scale2factor),
        v2.Lambda(pad2size),
    ])


def get_model_transforms(backbone="resnet18"):
    if backbone == "resnet18":
        return models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    else:
        raise ValueError("Unsupported backbone")


def extract_features(img, model, augment=0):
    preprocess = transforms.Compose(
        [
            get_static_transforms(),
            # v2.Lambda(ImageOps.autocontrast),
            # v2.Lambda(lambda x: ImageOps.equalize(x, mask=x.convert('L'))),
            get_model_transforms(),
        ]
    )
    preprocess_aug = transforms.Compose(
        [
            get_static_transforms(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomAffine(degrees=180, translate=(0.1, 0.1)),
            get_model_transforms(),
        ]
    )
    img_tensor = preprocess(img).unsqueeze(0)
    if augment > 0:
        aug_img_tensors = [preprocess_aug(
            img).unsqueeze(0) for _ in range(augment)]
        img_tensor = torch.cat([img_tensor, ] + aug_img_tensors, 0)
    model.eval()
    with torch.no_grad():
        outLinear, outDistance = model(img_tensor.cuda())
    return torch.median(outLinear, dim=0, keepdim=True).values, torch.median(outDistance, dim=0, keepdim=True).values
