import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class BasicBlock3D(nn.Module):
    """3D BasicBlock with configurable temporal inflation."""
    expansion = 1

    def __init__(self, in_planes, planes, spatial_stride=1, inflate=False):
        super(BasicBlock3D, self).__init__()
        self.inflate = inflate

        if inflate:
            # 3D Spatio-Temporal Convolution (res4, res5)
            conv1_kernel = (3, 3, 3)
            conv1_padding = (1, 1, 1)
            conv2_kernel = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            # 2D Spatial-only Convolution (conv1, res2, res3)
            conv1_kernel = (1, 3, 3)
            conv1_padding = (0, 1, 1)
            conv2_kernel = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        # In Table 2, all temporal strides are 1 (no temporal downsampling)
        stride = (1, spatial_stride, spatial_stride)

        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=conv1_kernel,
            stride=stride, padding=conv1_padding, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=conv2_kernel,
            stride=(1, 1, 1), padding=conv2_padding, bias=False
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = None
        if spatial_stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_planes, planes, kernel_size=(1, 1, 1),
                    stride=stride, bias=False
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


class ResNet3D18(nn.Module):
    """
    3D-ResNet-18 Backbone and Classifier Head as specified in Table 2 of F2-FixMatch:
    Stage   Layer                   Output Size (for 3x8x224x224)
    conv1   1x7x7, 64, stride 1,2,2 64 x 8 x 112 x 112
    pool1   1x3x3 max, stride 1,2,2 64 x 8 x 56 x 56
    res2    [1x3x3, 64; 1x3x3, 64]x2 64 x 8 x 56 x 56
    res3    [1x3x3, 128; 1x3x3, 128]x2 128 x 8 x 28 x 28  (stride 1,2,2)
    res4    [3x3x3, 256; 3x3x3, 256]x2 256 x 8 x 14 x 14  (stride 1,2,2, inflated)
    res5    [3x3x3, 512; 3x3x3, 512]x2 512 x 8 x 7 x 7    (stride 1,2,2, inflated)
    head    AdaptiveAvgPool3d(1), Dropout(0.5), FC(512, num_classes)
    """
    def __init__(self, num_classes=101, dropout_ratio=0.5):
        super(ResNet3D18, self).__init__()
        self.in_planes = 64

        # conv1: 1x7x7 kernel, stride (1, 2, 2)
        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=(1, 7, 7),
            stride=(1, 2, 2), padding=(0, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        # pool1: 1x3x3 max pool, stride (1, 2, 2)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        # res2: 2 blocks, 64 channels, 2D kernels
        self.layer1 = self._make_layer(64, num_blocks=2, spatial_stride=1, inflate=False)
        # res3: 2 blocks, 128 channels, 2D kernels, spatial stride 2
        self.layer2 = self._make_layer(128, num_blocks=2, spatial_stride=2, inflate=False)
        # res4: 2 blocks, 256 channels, 3D kernels, spatial stride 2
        self.layer3 = self._make_layer(256, num_blocks=2, spatial_stride=2, inflate=True)
        # res5: 2 blocks, 512 channels, 3D kernels, spatial stride 2
        self.layer4 = self._make_layer(512, num_blocks=2, spatial_stride=2, inflate=True)

        # Classifier head for 3D spatio-temporal features
        self.avgpool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Classifier head for 2D TAP (Temporal-Avg-Pooled) features
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, planes, num_blocks, spatial_stride, inflate):
        layers = []
        layers.append(BasicBlock3D(self.in_planes, planes, spatial_stride=spatial_stride, inflate=inflate))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock3D(self.in_planes, planes, spatial_stride=1, inflate=inflate))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_feat(self, x):
        """Extract high-level feature representation Z in R^{B x 512 x T x H' x W'}."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_3d_feat(self, feat):
        """Pass 3D feature through pooling, dropout, and FC head."""
        pooled = self.avgpool3d(feat).flatten(1)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits

    def forward_tap_feat(self, feat_2d):
        """Pass 2D TAP feature through pooling, dropout, and FC head."""
        pooled = self.avgpool2d(feat_2d).flatten(1)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits

    def forward(self, x):
        """Standard end-to-end forward pass."""
        feat = self.extract_feat(x)
        return self.forward_3d_feat(feat)


class FeaturePerturbator(nn.Module):
    """
    Feature-level Perturbator A_feat for Progressive Augmentation (P-Aug):
    1. Temporal-Avg-Pooling (TAP): averages over temporal dimension T (Z -> R^{B x C x H x W}).
    2. Spatial Dropout: randomly zeroes entire channels/patches with probability 0.5.
    3. Noise Injection: CutMix / patch swapping at feature level between batch samples.
    4. Feature Flipping: randomly flips features horizontally.
    """
    def __init__(self, spatial_drop_prob=0.5, flip_prob=0.5, noise_injection_prob=0.5):
        super(FeaturePerturbator, self).__init__()
        self.spatial_drop_prob = spatial_drop_prob
        self.flip_prob = flip_prob
        self.noise_injection_prob = noise_injection_prob

    def temporal_avg_pool(self, feat):
        """TAP: average along temporal dimension (dim=2). B x C x T x H x W -> B x C x H x W."""
        return feat.mean(dim=2)

    def apply_spatial_dropout(self, feat_2d):
        """Spatial dropout randomly drops channels/features."""
        if self.spatial_drop_prob > 0 and self.training:
            # Using 2D dropout treats channels as independent features across spatial map
            return F.dropout2d(feat_2d, p=self.spatial_drop_prob, training=self.training)
        return feat_2d

    def apply_feature_flipping(self, feat_2d):
        """Feature horizontal flip with flip_prob."""
        if random.random() < self.flip_prob and self.training:
            return torch.flip(feat_2d, dims=[-1])
        return feat_2d

    def apply_noise_injection(self, feat_2d):
        """Feature CutMix: inject a spatial patch from a randomly permuted sample."""
        if self.noise_injection_prob > 0 and feat_2d.size(0) > 1 and self.training and random.random() < self.noise_injection_prob:
            b, c, h, w = feat_2d.size()
            rand_indices = torch.randperm(b, device=feat_2d.device)
            # Choose a patch size (e.g. 2x2 to 3x3 on a 7x7 grid)
            patch_h = max(1, random.randint(1, max(1, h // 2)))
            patch_w = max(1, random.randint(1, max(1, w // 2)))
            top = random.randint(0, h - patch_h)
            left = random.randint(0, w - patch_w)

            perturbed = feat_2d.clone()
            perturbed[:, :, top:top+patch_h, left:left+patch_w] = feat_2d[rand_indices, :, top:top+patch_h, left:left+patch_w]
            return perturbed
        return feat_2d

    def forward(self, feat_3d):
        """Input: 3D feature Z (B x C x T x H x W). Output: Perturbed TAP feature (B x C x H x W)."""
        feat_2d = self.temporal_avg_pool(feat_3d)
        feat_2d = self.apply_spatial_dropout(feat_2d)
        feat_2d = self.apply_feature_flipping(feat_2d)
        feat_2d = self.apply_noise_injection(feat_2d)
        return feat_2d


class F2FixMatchModel(nn.Module):
    """
    Complete F2-FixMatch Network combining:
    - Shared 3D-ResNet18 Backbone for RGB and Temporal Gradient (TG) modalities
    - Feature Perturbator (P-Aug)
    - Classification Heads for 3D and TAP representations
    """
    def __init__(self, num_classes=101, dropout_ratio=0.5):
        super(F2FixMatchModel, self).__init__()
        self.backbone = ResNet3D18(num_classes=num_classes, dropout_ratio=dropout_ratio)
        self.perturbator = FeaturePerturbator()

    def forward_backbone(self, x):
        return self.backbone.extract_feat(x)

    def forward_logits(self, feat_3d):
        return self.backbone.forward_3d_feat(feat_3d)

    def forward_paug(self, feat_3d):
        """Apply P-Aug and compute predictions on perturbed feature."""
        feat_p = self.perturbator(feat_3d)
        logits_p = self.backbone.forward_tap_feat(feat_p)
        return feat_p, logits_p

    def forward_eval(self, x):
        """Evaluation forward pass (RGB only)."""
        return self.backbone(x)
