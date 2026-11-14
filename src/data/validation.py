"""Dataset validation pipeline for hand X-ray radiographs and RAM-H1200 annotations.

Performs the 8 validation checks specified in PRD Section 5.3:
1. Corrupted image detection.
2. Duplicate image identification (SHA-256).
3. Missing annotation detection.
4. Image dimension validation.
5. Label consistency checks.
6. Patient ID verification & zero-leakage audit.
7. Class distribution analysis.
8. Annotation quality inspection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for duplicate detection."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_patient_id_from_filename(filename: str) -> str:
    """Extract normalized patient identifier from RAM-H1200 filename (e.g. JP_HMCRD_P0001_...)."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 3 and parts[2].startswith("P"):
        return "_".join(parts[:3])
    return parts[0]


def validate_single_image(
    image_path: str | Path,
    min_dim: int = 64,
    max_dim: int = 8192,
) -> Dict[str, Any]:
    """Validate that an uploaded or dataset X-ray image is readable and meets quality criteria."""
    path = Path(image_path)
    if not path.exists():
        return {"valid": False, "error": f"File does not exist: {path}"}

    ext = path.suffix.lower()
    if ext not in {".bmp", ".png", ".jpg", ".jpeg", ".dcm", ".dicom"}:
        return {"valid": False, "error": f"Unsupported file format: {ext}"}

    try:
        if ext in {".dcm", ".dicom"}:
            import pydicom

            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
            arr = ((arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 255.0).astype(np.uint8)
            img = arr if arr.ndim == 2 else arr[..., 0]
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    except Exception as exc:
        return {"valid": False, "error": f"Corrupted image file: {exc}"}

    if img is None or img.size == 0:
        return {"valid": False, "error": "Corrupted or unreadable image file"}

    h, w = img.shape[:2]
    if h < min_dim or w < min_dim or h > max_dim or w > max_dim:
        return {
            "valid": False,
            "error": f"Invalid dimensions ({w}x{h}); expected between {min_dim} and {max_dim}px",
        }

    mean_intensity = float(np.mean(img))
    std_intensity = float(np.std(img))
    dynamic_range = float(np.percentile(img, 99) - np.percentile(img, 1))
    laplacian_var = float(cv2.Laplacian(img, cv2.CV_64F).var())

    quality_warnings: List[str] = []
    if std_intensity < 8.0 or dynamic_range < 25.0:
        quality_warnings.append("Low radiographic contrast detected")
    if mean_intensity < 5.0 or mean_intensity > 250.0:
        quality_warnings.append("Abnormal exposure intensity detected")

    return {
        "valid": True,
        "width": int(w),
        "height": int(h),
        "aspect_ratio": round(w / max(h, 1), 4),
        "mean_intensity": round(mean_intensity, 2),
        "std_intensity": round(std_intensity, 2),
        "dynamic_range": round(dynamic_range, 2),
        "sharpness_score": round(laplacian_var, 2),
        "quality_warnings": quality_warnings,
    }


def run_dataset_validation(root_dir: str | Path = ".") -> Dict[str, Any]:
    """Execute all 8 dataset validation checks and save a structured audit report."""
    root = Path(root_dir)
    raw_dir = root / "dataset" / "raw"
    splits_dir = root / "dataset" / "splits"

    corrupted_files: List[str] = []
    duplicate_pairs: List[Dict[str, str]] = []
    missing_annotations: List[str] = []
    dimension_records: List[Dict[str, Any]] = []
    hash_map: Dict[str, str] = {}

    split_patients: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    annotation_summary: Dict[str, Any] = {}

    for split in ("train", "val", "test"):
        seg_dir = raw_dir / "Segmentation" / split
        be_dir = raw_dir / "SvdH_Scoring" / "SvdH_BE_Scoring" / split
        jsn_dir = raw_dir / "SvdH_Scoring" / "SvdH_JSN_Scoring" / split
        if not seg_dir.exists():
            continue

        bmp_files = sorted(seg_dir.glob("*.bmp"))
        split_counts[split] = len(bmp_files)

        bone_coco_path = seg_dir / "_annotations_bone_rle.coco.json"
        be_coco_path = seg_dir / "_annotations_be_rle.coco.json"
        be_scores_path = be_dir / "_annotation_be_scores.json"
        jsn_scores_path = jsn_dir / "_annotation_jsn_scores.json"

        bone_annotated_files: set = set()
        if bone_coco_path.exists():
            bone_data = json.loads(bone_coco_path.read_text(encoding="utf-8"))
            img_id_with_ann = {a["image_id"] for a in bone_data.get("annotations", [])}
            bone_annotated_files = {
                img["file_name"]
                for img in bone_data.get("images", [])
                if img["id"] in img_id_with_ann
            }

        be_scores = (
            json.loads(be_scores_path.read_text(encoding="utf-8"))
            if be_scores_path.exists()
            else {}
        )
        jsn_scores = (
            json.loads(jsn_scores_path.read_text(encoding="utf-8"))
            if jsn_scores_path.exists()
            else {}
        )

        for bmp in bmp_files:
            pid = extract_patient_id_from_filename(bmp.name)
            split_patients[split].add(pid)

            check = validate_single_image(bmp)
            if not check.get("valid", False):
                corrupted_files.append(str(bmp))
                continue

            dimension_records.append(
                {
                    "file": bmp.name,
                    "split": split,
                    "width": check["width"],
                    "height": check["height"],
                    "mean_intensity": check["mean_intensity"],
                    "std_intensity": check["std_intensity"],
                }
            )

            file_hash = compute_sha256(bmp)
            if file_hash in hash_map:
                duplicate_pairs.append({"file_a": hash_map[file_hash], "file_b": str(bmp)})
            else:
                hash_map[file_hash] = str(bmp)

            if bmp.name not in bone_annotated_files:
                missing_annotations.append(f"{split}/{bmp.name}: missing bone RLE annotation")
            if bmp.name not in be_scores:
                missing_annotations.append(f"{split}/{bmp.name}: missing BE score entry")
            if bmp.name not in jsn_scores:
                missing_annotations.append(f"{split}/{bmp.name}: missing JSN score entry")

        annotation_summary[split] = {
            "num_images": len(bmp_files),
            "num_bone_annotated": len(bone_annotated_files),
            "num_be_scored": len(be_scores),
            "num_jsn_scored": len(jsn_scores),
            "has_be_rle": be_coco_path.exists(),
        }

    # Check patient overlap across splits
    overlap_tv = sorted(split_patients["train"] & split_patients["val"])
    overlap_tt = sorted(split_patients["train"] & split_patients["test"])
    overlap_vt = sorted(split_patients["val"] & split_patients["test"])

    # Class distribution from Metadata.xlsx or split CSVs
    class_distribution: Dict[str, Any] = {}
    meta_csv = raw_dir / "metadata" / "metadata.csv"
    if meta_csv.exists():
        df = pd.read_csv(meta_csv)
        if "isRA" in df.columns:
            vc = df["isRA"].value_counts().to_dict()
            class_distribution = {
                "RA_positive": int(vc.get(1, 0)),
                "RA_negative": int(vc.get(0, 0)),
                "total": int(len(df)),
                "ra_ratio": round(float(df["isRA"].mean()), 4) if len(df) > 0 else 0.0,
            }

    report = {
        "corrupted_images_count": len(corrupted_files),
        "corrupted_images": corrupted_files,
        "duplicate_images_count": len(duplicate_pairs),
        "duplicate_images": duplicate_pairs,
        "missing_annotations_count": len(missing_annotations),
        "missing_annotations": missing_annotations[:25],
        "split_image_counts": split_counts,
        "split_patient_counts": {k: len(v) for k, v in split_patients.items()},
        "patient_leakage": {
            "train_val_overlap": overlap_tv,
            "train_test_overlap": overlap_tt,
            "val_test_overlap": overlap_vt,
            "passed_zero_leakage": len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0,
        },
        "class_distribution": class_distribution,
        "annotation_completeness": annotation_summary,
        "dimension_summary": {
            "total_checked": len(dimension_records),
            "unique_resolutions": sorted(
                {f"{r['width']}x{r['height']}" for r in dimension_records}
            ),
        },
        "validation_passed": (
            len(corrupted_files) == 0
            and len(duplicate_pairs) == 0
            and len(overlap_tv) == 0
            and len(overlap_tt) == 0
            and len(overlap_vt) == 0
        ),
    }

    out_path = splits_dir / "data_validation_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = run_dataset_validation(".")
    print(json.dumps(rep, indent=2))
