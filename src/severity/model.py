"""Multi-head SvdH Bone Erosion (BE) and Joint Space Narrowing (JSN) Scoring Network."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchvision import models


class SvdHSeverityNet(nn.Module):
    """Predicts joint-level SvdH Bone Erosion (16 joints, 0-5) and JSN (15 joints, 0-4) scores."""

    def __init__(
        self,
        num_be_joints: int = 16,
        num_jsn_joints: int = 15,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.architecture = "svdh_multitask_resnet18"
        self.num_be_joints = num_be_joints
        self.num_jsn_joints = num_jsn_joints

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.shared_proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )
        self.be_head = nn.Linear(256, num_be_joints)
        self.jsn_head = nn.Linear(256, num_jsn_joints)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.shared_proj(self.backbone(x))
        # Constrain BE per joint to [0, 5] and JSN per joint to [0, 4]
        be_scores = torch.sigmoid(self.be_head(feats)) * 5.0
        jsn_scores = torch.sigmoid(self.jsn_head(feats)) * 4.0
        return {
            "be_scores": be_scores,
            "jsn_scores": jsn_scores,
            "total_be": be_scores.sum(dim=1),
            "total_jsn": jsn_scores.sum(dim=1),
        }
