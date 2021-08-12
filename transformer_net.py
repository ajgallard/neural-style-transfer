"""
Script for building the fast neural style transfer architecture
A visual of the following code is provided as a png taken from
an article 'Neural Style Transfer: Applications in Data Augmentation'
by Conner Shorten
Source for Code Acknowledgements: Pytorch, rrmina
Further acknowledgements for code in the repo README file
"""

import torch.nn as nn


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Encoder Layer
        self.Encoder = nn.Sequential(
            ConvLayer(3, 32, 9, 1), nn.ReLU(),
            ConvLayer(32, 64, 3, 2), nn.ReLU(),
            ConvLayer(64, 128, 3, 2), nn.ReLU(),
        )
        # Style Bank (AKA Residual Block)
        self.StyleBank = nn.Sequential(
            ResidualLayer(),
            ResidualLayer(),
            ResidualLayer(),
            ResidualLayer(),
            ResidualLayer(),
        )
        # Decoder
        self.Decoder = nn.Sequential(
            UpsampleConvLayer(128, 64, 3, 1, 2),  nn.ReLU(),
            UpsampleConvLayer(64, 32, 3, 1, 2), nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm='None'),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.StyleBank(x)
        out = self.Decoder(x)
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm='instance'):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if (norm == 'instance'):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x1 = self.reflection_pad(x)
        x2 = self.conv2d(x1)
        if (self.norm_type == 'None'):
            out = x2
        else:
            out = self.norm_layer(x2)
        return out


class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, stride=1):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(nn.Module):
    """Applying Upsampling rather than ConvTranspose based on Pytorch Example Code"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
