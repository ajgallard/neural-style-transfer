"""
Building partial Vgg16 with torchvision
This will allow us to train the model on the style image
and use the pretrained weights from Vgg16 for the content image
"""

from collections import namedtuple
import torch.nn as nn
from torchvision import models

# using Vgg16
class VggNet(nn.Module):
    def __init__(self, requires_grad=False):
        super(VggNet, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features)
        for x in range(4,9):
            self.slice2.add_module(str(x), vgg_pretrained_features)
        for x in range(9,16):
            self.slice3.add_module(str(x), vgg_pretrained_features)
        for x in range(16,23):
            self.slice4.add_module(str(x), vgg_pretrained_features)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice1(x)
        relu1_2 = x
        x = self.slice2(x)
        relu2_2 = x
        x = self.slice3(x)
        relu3_3 = x
        x = self.slice4(x)
        relu4_3 = x
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(relu1_2, relu2_2, relu3_3, relu4_3)
        return out
