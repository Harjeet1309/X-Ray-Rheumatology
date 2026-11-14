"""Inference engine for Model 4: SvdH Radiographic Severity Assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from src.data.generate_sample_fixtures import BE_JOINTS, JSN_JOINTS
from src.data.preprocessing import preprocess_xray
from src.severity.model import SvdHSeverityNet


class SeverityInferenceService:
    """Predicts joint-level and total hand Sharp/van der Heijde (SvdH) BE & JSN scores."""

    def __init__(self, root_dir: str | Path = ".") -> None:
        self.root = Path(root_dir)
        self.ckpt_path = self.root / "models" / "severity" / "best_model.pth"
        self.version = "1.0.0"
        self.architecture = "svdh_multitask_resnet18"
        self.image_size: Tuple[int, int] = (128, 128)
        self.model = SvdHSeverityNet(
            num_be_joints=len(BE_JOINTS),
            num_jsn_joints=len(JSN_JOINTS),
            pretrained=False,
        )
        self._load()

    def _load(self) -> None:
        if self.ckpt_path.exists():
            ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            self.version = ckpt.get("version", "1.0.0")
            self.architecture = ckpt.get("architecture", "svdh_multitask_resnet18")
            self.image_size = tuple(ckpt.get("image_size", [128, 128]))
            self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(self, image_or_path: str | Path | np.ndarray) -> Dict[str, Any]:
        """Predict joint-level BE (0-5) and JSN (0-4) SvdH scores and aggregate hand totals."""
        prep = preprocess_xray(
            image_or_path, target_size=self.image_size, use_clahe=True, three_channel=False
        )
        norm_2d = prep["normalized"]
        stacked = np.stack([norm_2d, norm_2d, norm_2d], axis=0).astype(np.float32)
        tensor = torch.from_numpy(stacked).unsqueeze(0)

        with torch.no_grad():
            out = self.model(tensor)
            be_raw = out["be_scores"][0].cpu().numpy()
            jsn_raw = out["jsn_scores"][0].cpu().numpy()

        be_joint_scores: Dict[str, int] = {}
        for j_name, val in zip(BE_JOINTS, be_raw):
            be_joint_scores[j_name] = int(np.clip(np.round(val), 0, 5))

        jsn_joint_scores: Dict[str, int] = {}
        for j_name, val in zip(JSN_JOINTS, jsn_raw):
            jsn_joint_scores[j_name] = int(np.clip(np.round(val), 0, 4))

        total_be = int(sum(be_joint_scores.values()))
        total_jsn = int(sum(jsn_joint_scores.values()))
        total_svdh = total_be + total_jsn

        if total_svdh <= 6:
            category = "Minimal / Remission-Range Structural Change"
        elif total_svdh <= 24:
            category = "Mild Radiographic Structural Damage"
        elif total_svdh <= 55:
            category = "Moderate Radiographic Structural Damage"
        else:
            category = "Severe Radiographic Structural Damage"

        return {
            "total_be_score": total_be,
            "max_be_score": 80,
            "total_jsn_score": total_jsn,
            "max_jsn_score": 60,
            "total_svdh_score": total_svdh,
            "max_svdh_score": 140,
            "severity_category": category,
            "be_joint_scores": be_joint_scores,
            "jsn_joint_scores": jsn_joint_scores,
            "scoring_protocol": "Sharp/van der Heijde (SvdH) Single-Hand Protocol (16 BE + 15 JSN joints)",
            "uncertainty_note": (
                "Estimated from single-view radiograph using RAM-H1200 reference annotations. "
                "Requires clinical confirmation by a qualified rheumatologist or radiologist."
            ),
            "architecture": self.architecture,
            "model_version": self.version,
        }
