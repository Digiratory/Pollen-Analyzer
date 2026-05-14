import torch
from torch import nn
from PIL import ImageOps
import torchvision.transforms as transforms
from torchvision.transforms import v2 as T
from common.img_utils import pad2size, scale2factor


class CloseSetClassificationModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.num_classes = n_classes
        self.model: nn.Module = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.model.eval()
        num_hidden_features = 384

        self.linear = nn.Linear(num_hidden_features, n_classes, bias=False)
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


    def train(self, mode: bool):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x, skip_distance=False):
        batch_size = len(x)
        x = self.model(x)
        x = x.view(batch_size, -1)
        outLinear = self.classifier(x)

        return outLinear, None