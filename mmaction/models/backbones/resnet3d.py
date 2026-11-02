"""
3D-ResNet-18 Backbone matching MMAction2 conventions and Table 2 in F2-FixMatch.
Spatio-temporal convolution architecture with 2D spatial early layers (res2, res3)
and inflated 3D layers (res4, res5) totaling 31.87M parameters.
"""

import torch
import torch.nn as nn
from ..builder import BACKBONES


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, spatial_stride=1, inflate=False):
        super(BasicBlock3d, self).__init__()
        self.inflate = inflate

        if inflate:
            kernel_size = (3, 3, 3)
            padding = (1, 1, 1)
            stride = (1, spatial_stride, spatial_stride)
        else:
            kernel_size = (1, 3, 3)
            padding = (0, 1, 1)
            stride = (1, spatial_stride, spatial_stride)

        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=kernel_size,
            stride=(1, 1, 1), padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = None
        if spatial_stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_planes, planes, kernel_size=1,
                    stride=(1, spatial_stride, spatial_stride), bias=False
                ),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


@BACKBONES.register_module()
class ResNet3d(nn.Module):
    """
    3D-ResNet backbone for MMAction2.
    Depth 18 configuration matching Table 2 of F2-FixMatch paper.
    """

    def __init__(self,
                 depth=18,
                 pretrained=None,
                 pretrained2d=False,
                 in_channels=3,
                 base_channels=64,
                 out_indices=(3,),
                 spatial_strides=(1, 2, 2, 2),
                 inflate=(0, 0, 1, 1),
                 **kwargs):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.out_indices = out_indices
        self.in_planes = base_channels

        # Stem: conv1 (1x7x7, stride 1, 2, 2)
        self.conv1 = nn.Conv3d(
            in_channels, base_channels, kernel_size=(1, 7, 7),
            stride=(1, 2, 2), padding=(0, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        # pool1: 1x3x3 max pool, stride (1, 2, 2)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        # Stages: res2 (64), res3 (128), res4 (256), res5 (512)
        self.layer1 = self._make_layer(64, num_blocks=2, spatial_stride=spatial_strides[0], inflate=bool(inflate[0]))
        self.layer2 = self._make_layer(128, num_blocks=2, spatial_stride=spatial_strides[1], inflate=bool(inflate[1]))
        self.layer3 = self._make_layer(256, num_blocks=2, spatial_stride=spatial_strides[2], inflate=bool(inflate[2]))
        self.layer4 = self._make_layer(512, num_blocks=2, spatial_stride=spatial_strides[3], inflate=bool(inflate[3]))

        self.init_weights()

    def _make_layer(self, planes, num_blocks, spatial_stride, inflate):
        layers = []
        layers.append(BasicBlock3d(self.in_planes, planes, spatial_stride=spatial_stride, inflate=inflate))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock3d(self.in_planes, planes, spatial_stride=1, inflate=inflate))
        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """Input: (B, C, T, H, W). Output: (B, 512, T, H', W')."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
