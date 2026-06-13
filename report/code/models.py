"""Model definitions: shallow CNN baseline + transfer-learning wrappers."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class ShallowCNN(nn.Module):
    """A small CNN trained from scratch as a baseline.

    ~0.5M parameters. Three conv blocks, global average pool, linear head.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    net = models.efficientnet_b0(weights=weights)
    in_f = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_f, num_classes)
    return net


def build_mobilenet_v3_small(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    in_f = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_f, num_classes)
    return net


REGISTRY = {
    "ShallowCNN": lambda: ShallowCNN(2),
    "ResNet-18": lambda: build_resnet18(2, pretrained=True),
    "EfficientNet-B0": lambda: build_efficientnet_b0(2, pretrained=True),
    "MobileNet-V3-Small": lambda: build_mobilenet_v3_small(2, pretrained=True),
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
