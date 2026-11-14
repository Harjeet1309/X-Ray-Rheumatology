"""Candidate deep-learning architectures for RA Classification (PRD Section 8.1).

Supports:
- efficientnet_b0
- efficientnet_b3
- densenet121
- resnet50
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class RAClassifier(nn.Module):
    """Transfer-learning classifier for hand X-ray RA pattern recognition."""

    def __init__(
        self,
        architecture: str = "densenet121",
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes

        if self.architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.target_layer = self.backbone.layer4[-1]
        elif self.architecture == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
            self.target_layer = self.backbone.features[-1]
        elif self.architecture == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            backbone = models.efficientnet_b3(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
            self.target_layer = self.backbone.features[-1]
        else:
            # Default: densenet121
            self.architecture = "densenet121"
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            backbone = models.densenet121(weights=weights)
            in_features = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            self.backbone = backbone
            self.target_layer = self.backbone.features.denseblock4

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    def get_target_conv_layer(self) -> nn.Module:
        """Return target convolutional block for Grad-CAM visualization."""
        return self.target_layer


def build_classifier(
    architecture: str = "densenet121",
    num_classes: int = 2,
    pretrained: bool = False,
) -> RAClassifier:
    """Factory function to instantiate a candidate RA classification model."""
    return RAClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
    )
