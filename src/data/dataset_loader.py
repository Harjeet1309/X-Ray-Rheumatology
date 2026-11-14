"""PyTorch Dataset and DataLoader builders for all four RAM-H1200 tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.augmentation import RadiographAugmentor
from src.data.generate_sample_fixtures import BE_JOINTS, JSN_JOINTS, decode_rle_to_mask
from src.data.preprocessing import letterbox_resize, preprocess_xray


class RAMH1200ClassificationDataset(Dataset):
    """Dataset for Model 1: Rheumatoid Arthritis Classification (`isRA`)."""

    def __init__(
        self,
        split_csv: str | Path,
        root_dir: str | Path = ".",
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
    ) -> None:
        self.root = Path(root_dir)
        self.df = pd.read_csv(split_csv)
        self.image_size = image_size
        self.augmentor = RadiographAugmentor() if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self.root / str(row["image_path"])
        out = preprocess_xray(img_path, target_size=self.image_size, three_channel=False)
        norm_img = out["normalized"]
        if self.augmentor is not None:
            norm_img, _ = self.augmentor(norm_img)

        stacked = np.stack([norm_img, norm_img, norm_img], axis=0).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        tensor = torch.from_numpy((stacked - mean) / std)

        return {
            "image": tensor,
            "label": torch.tensor(int(row["is_ra"]), dtype=torch.long),
            "patient_id": str(row["patient_id"]),
            "image_stem": str(row["image_stem"]),
        }


class RAMH1200SegmentationDataset(Dataset):
    """Dataset for Model 3: Bone and Erosion Pixel-Level Segmentation."""

    def __init__(
        self,
        split_name: str = "train",
        root_dir: str | Path = ".",
        image_size: Tuple[int, int] = (256, 256),
        task: str = "combined",  # "bone", "erosion", or "combined"
        augment: bool = False,
    ) -> None:
        self.root = Path(root_dir)
        self.split_name = split_name
        self.image_size = image_size
        self.task = task
        self.augmentor = RadiographAugmentor() if augment else None

        seg_dir = self.root / "dataset" / "raw" / "Segmentation" / split_name
        self.bmp_files = sorted(seg_dir.glob("*.bmp"))

        bone_json = seg_dir / "_annotations_bone_rle.coco.json"
        be_json = seg_dir / "_annotations_be_rle.coco.json"
        self.bone_coco = (
            json.loads(bone_json.read_text(encoding="utf-8")) if bone_json.exists() else {}
        )
        self.be_coco = (
            json.loads(be_json.read_text(encoding="utf-8")) if be_json.exists() else {}
        )

    def __len__(self) -> int:
        return len(self.bmp_files)

    def _build_mask_from_coco(
        self, coco_dict: Dict[str, Any], file_name: str, h: int, w: int
    ) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        if not coco_dict:
            return mask
        img_id = None
        for im in coco_dict.get("images", []):
            if im["file_name"] == file_name:
                img_id = im["id"]
                break
        if img_id is None:
            return mask
        for ann in coco_dict.get("annotations", []):
            if ann["image_id"] == img_id and "segmentation" in ann:
                bin_m = decode_rle_to_mask(ann["segmentation"])
                if bin_m.shape == (h, w):
                    mask[bin_m > 0] = int(ann.get("category_id", 1))
        return mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        bmp_path = self.bmp_files[idx]
        prep = preprocess_xray(bmp_path, target_size=self.image_size, three_channel=False)
        img_norm = prep["normalized"]
        orig_h = prep["meta"]["orig_height"]
        orig_w = prep["meta"]["orig_width"]

        bone_m = self._build_mask_from_coco(self.bone_coco, bmp_path.name, orig_h, orig_w)
        be_m = self._build_mask_from_coco(self.be_coco, bmp_path.name, orig_h, orig_w)

        if self.task == "bone":
            # 0=bg, 1..5=bone classes -> collapse into 0=bg, 1=cortical bone, 2=carpal, 3=phalanx
            target_m = np.clip(bone_m, 0, 3).astype(np.uint8)
        elif self.task == "erosion":
            target_m = np.clip(be_m, 0, 3).astype(np.uint8)
        else:
            # 4-class clinically actionable map: 0=Background, 1=Bone, 2=Carpal/Joint, 3=Bone Erosion
            target_m = np.zeros_like(bone_m, dtype=np.uint8)
            target_m[bone_m > 0] = 1
            target_m[bone_m == 3] = 2
            target_m[be_m > 0] = 3

        target_resized, _ = letterbox_resize(target_m, target_size=self.image_size, is_mask=True)
        if self.augmentor is not None:
            img_norm, target_resized = self.augmentor(img_norm, target_resized)

        return {
            "image": torch.from_numpy(img_norm[None, ...]).float(),
            "mask": torch.from_numpy(target_resized).long(),
            "file_name": bmp_path.name,
        }


class RAMH1200SeverityDataset(Dataset):
    """Dataset for Model 4: SvdH Joint-Level and Hand-Level BE & JSN Severity Scoring."""

    def __init__(
        self,
        split_name: str = "train",
        root_dir: str | Path = ".",
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
    ) -> None:
        self.root = Path(root_dir)
        self.split_name = split_name
        self.image_size = image_size
        self.augmentor = RadiographAugmentor() if augment else None

        seg_dir = self.root / "dataset" / "raw" / "Segmentation" / split_name
        be_score_file = (
            self.root
            / "dataset"
            / "raw"
            / "SvdH_Scoring"
            / "SvdH_BE_Scoring"
            / split_name
            / "_annotation_be_scores.json"
        )
        jsn_score_file = (
            self.root
            / "dataset"
            / "raw"
            / "SvdH_Scoring"
            / "SvdH_JSN_Scoring"
            / split_name
            / "_annotation_jsn_scores.json"
        )

        self.bmp_files = sorted(seg_dir.glob("*.bmp"))
        self.be_scores = (
            json.loads(be_score_file.read_text(encoding="utf-8"))
            if be_score_file.exists()
            else {}
        )
        self.jsn_scores = (
            json.loads(jsn_score_file.read_text(encoding="utf-8"))
            if jsn_score_file.exists()
            else {}
        )

    def __len__(self) -> int:
        return len(self.bmp_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        bmp_path = self.bmp_files[idx]
        prep = preprocess_xray(bmp_path, target_size=self.image_size, three_channel=False)
        img_norm = prep["normalized"]
        if self.augmentor is not None:
            img_norm, _ = self.augmentor(img_norm)

        stacked = np.stack([img_norm, img_norm, img_norm], axis=0).astype(np.float32)

        be_entry = self.be_scores.get(bmp_path.name, {})
        jsn_entry = self.jsn_scores.get(bmp_path.name, {})

        be_vec = [float(be_entry.get(f"BE_{j}", 0)) for j in BE_JOINTS]
        jsn_vec = [float(jsn_entry.get(f"JSN_{j}", 0)) for j in JSN_JOINTS]

        return {
            "image": torch.from_numpy(stacked).float(),
            "be_scores": torch.tensor(be_vec, dtype=torch.float32),
            "jsn_scores": torch.tensor(jsn_vec, dtype=torch.float32),
            "total_be": torch.tensor(sum(be_vec), dtype=torch.float32),
            "total_jsn": torch.tensor(sum(jsn_vec), dtype=torch.float32),
            "file_name": bmp_path.name,
        }


def create_dataloaders(
    root_dir: str | Path = ".",
    batch_size: int = 4,
    image_size: Tuple[int, int] = (256, 256),
) -> Dict[str, DataLoader]:
    """Create train/val/test DataLoaders for classification."""
    root = Path(root_dir)
    splits_dir = root / "dataset" / "splits"
    train_ds = RAMH1200ClassificationDataset(
        splits_dir / "train.csv", root_dir=root, image_size=image_size, augment=True
    )
    val_ds = RAMH1200ClassificationDataset(
        splits_dir / "validation.csv", root_dir=root, image_size=image_size, augment=False
    )
    test_ds = RAMH1200ClassificationDataset(
        splits_dir / "test.csv", root_dir=root, image_size=image_size, augment=False
    )
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }
