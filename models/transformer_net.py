"""
Script for building the fast neural style transfer architecture
A visual of the following code is provided as a png taken from
an article 'Neural Style Transfer: Applications in Data Augmentation'
by Conner Shorten
Further acknowledgements for code in the repo README file
"""

import torch.nn as nn

class TransformerNet(nn.Module):
    def __init__(self):
        super(FastNST, self).__init__()
        # Encoder Layer
        self.Encoder = nn.Sequential(
            ConvLayer(3, 32, 9, 1), nn.ReLU(),
            ConvLayer(32, 64, 3, 2), nn.ReLU(),
            ConvLayer(64, 128, 3, 2), nn.ReLU(),
        )

        # Style Bank (AKA Residual Block)
        self.StyleBank = nn.Sequential (
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
            ResidualLayer(128, 3),
        )

        # Decoder
        self.Decoder = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1), nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1), nn.ReLU(),
            DeconvLayer(32, 3, 9, 1), nn.ReLU(),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.StyleBank(x)
        out = self.Decoder(x)
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.instance_norm2d = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.instance_norm2d(out)
        return out

class ResidualLayer(nn.Module):
    def __init__(self, channels=128, kernel_size=3, stride=1):
        super(ResidualLayer, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out

class DeconvLayer(nn.Module):
    """Applying Upsampling rather than ConvTranspose based on Pytorch Example Code"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(DeconvLayer, self).__init__()
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
