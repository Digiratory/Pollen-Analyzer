import torch
from torch import nn
from PIL import ImageOps
import torchvision.transforms as transforms
from torchvision.transforms import v2 as T
from common.img_utils import pad2size, scale2factor


class OpenSetClassificationModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.num_classes = n_classes
        self.model: nn.Module = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.model.eval()
        num_hidden_features = 384
        layers = 1
        self.linear = nn.Linear((1 + layers) *num_hidden_features, n_classes, bias=False)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        # self.linear.bias.data.zero_()
        self.classifier = self.linear
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     nn.BatchNorm1d(num_features),
        #     nn.Linear(num_features, num_hidden_features, bias=True),
        #     nn.Mish(),
        #     nn.LayerNorm(num_hidden_features),
        #     nn.Linear(num_hidden_features, n_classes, bias=False)
        # )

        self.anchors = nn.Parameter(torch.zeros(
            self.num_classes, self.num_classes).float(), requires_grad=False)

    def train(self, mode: bool):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x, skip_distance=False):
        batch_size = len(x)
        x = self.model.forward_features(x)
        cls_token = x["x_norm_clstoken"]
        patch_tokens = x["x_norm_patchtokens"]
        # fmt: off
        linear_input = torch.cat([
            cls_token,
            patch_tokens.mean(dim=1),
        ], dim=1)
        # fmt: on
        # x = x.view(batch_size, -1)

        outLinear = self.classifier(linear_input)
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


def extract_features(img, model, augment=0):
    preprocess = transforms.Compose(
        [
            T.Lambda(scale2factor),
            T.Lambda(pad2size),
            # T.Lambda(ImageOps.autocontrast),
            # T.Lambda(lambda x: ImageOps.equalize(x, mask=x.convert('L'))),
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225]),
        ]
    )
    preprocess_aug = transforms.Compose(
        [
            T.Lambda(scale2factor),
            T.Lambda(pad2size),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomEqualize(),
            T.RandomVerticalFlip(p=0.5),
            # T.RandomRotation(180),
            T.RandomAffine(degrees=180, translate=(0.1, 0.1)),
            # T.RandomAutocontrast(),
            # T.Lambda(ImageOps.autocontrast),
            # T.Lambda(lambda x: ImageOps.equalize(x, mask=x.convert('L'))),
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225]),
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
