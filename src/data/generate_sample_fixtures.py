"""Generate a RAM-H1200-structured dataset fixture for immediate local training & testing.

Mirrors the exact directory layout, file naming convention, COCO RLE segmentation
annotations, COCO joint detection annotations, SvdH BE/JSN joint ROI folders,
and Metadata.xlsx schema of TokyoTechMagicYang/RAM-H1200-v1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

BE_JOINTS = [
    "MCP-T", "MCP-I", "MCP-M", "MCP-R", "MCP-S",
    "IP", "PIP-I", "PIP-M", "PIP-R", "PIP-S",
    "CMC-T", "Tm", "S", "L", "U", "R",
]

JSN_JOINTS = [
    "MCP-T", "MCP-I", "MCP-M", "MCP-R", "MCP-S",
    "PIP-I", "PIP-M", "PIP-R", "PIP-S",
    "STT", "SC", "SR", "CMC-M", "CMC-R", "CMC-S",
]

BONE_CATEGORIES = [
    {"id": 1, "name": "Radius"},
    {"id": 2, "name": "Ulna"},
    {"id": 3, "name": "Carpal"},
    {"id": 4, "name": "Metacarpal"},
    {"id": 5, "name": "Phalanx"},
]

BE_SEG_CATEGORIES = [
    {"id": 1, "name": "Non-SvdH-BE"},
    {"id": 2, "name": "SvdH-BE-50"},
    {"id": 3, "name": "SvdH-BE-90"},
]


def encode_binary_mask_to_uncompressed_rle(mask: np.ndarray) -> Dict[str, Any]:
    """Encode a 2D binary mask (H, W) into COCO uncompressed RLE format."""
    h, w = mask.shape
    flat = mask.T.flatten().astype(np.uint8)
    counts: List[int] = []
    current_val = 0
    run_len = 0
    for pixel in flat:
        if int(pixel) == current_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_len = 1
            current_val = int(pixel)
    counts.append(run_len)
    return {"size": [int(h), int(w)], "counts": counts}


def decode_rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Decode uncompressed or compressed COCO RLE into a 2D binary mask (H, W)."""
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, str):
        try:
            from pycocotools import mask as mask_utils
            return mask_utils.decode(rle).astype(np.uint8)
        except Exception:
            return np.zeros((h, w), dtype=np.uint8)
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if val == 1 and idx + c <= flat.size:
            flat[idx : idx + c] = 1
        idx += c
        val = 1 - val
    return flat.reshape((w, h)).T


def _synthesize_hand_xray(
    height: int,
    width: int,
    is_ra: bool,
    hand_side: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Tuple[int, int, int, int]]]:
    """Synthesize a realistic hand radiograph BMP with anatomical bones, joints, and erosions.

    Returns:
        image_u8: (H, W) uint8 radiograph
        bone_mask: (H, W) uint8 class mask (0=bg, 1=Radius, 2=Ulna, 3=Carpal, 4=Metacarpal, 5=Phalanx)
        erosion_mask: (H, W) uint8 class mask (0=bg, 1=Non-SvdH-BE, 2=SvdH-BE-50, 3=SvdH-BE-90)
        joint_boxes: dict mapping joint name -> (x, y, w, h)
    """
    canvas = np.full((height, width), 18.0, dtype=np.float32)
    bone_mask = np.zeros((height, width), dtype=np.uint8)
    erosion_mask = np.zeros((height, width), dtype=np.uint8)
    joint_boxes: Dict[str, Tuple[int, int, int, int]] = {}

    cx = width // 2 + int(rng.integers(-12, 13))
    wrist_y = int(height * 0.74)

    # Soft tissue envelope
    cv2.ellipse(
        canvas,
        (cx, int(height * 0.56)),
        (int(width * 0.28), int(height * 0.34)),
        0,
        0,
        360,
        46.0,
        -1,
    )

    # Radius and Ulna
    rad_x = cx - 28 if hand_side == "L" else cx + 28
    uln_x = cx + 28 if hand_side == "L" else cx - 28
    cv2.rectangle(canvas, (rad_x - 22, wrist_y), (rad_x + 22, height - 20), 175.0, -1)
    cv2.rectangle(bone_mask, (rad_x - 22, wrist_y), (rad_x + 22, height - 20), 1, -1)
    cv2.rectangle(canvas, (uln_x - 16, wrist_y + 6), (uln_x + 16, height - 20), 168.0, -1)
    cv2.rectangle(bone_mask, (uln_x - 16, wrist_y + 6), (uln_x + 16, height - 20), 2, -1)

    joint_boxes["R"] = (rad_x - 26, wrist_y - 18, 52, 36)
    joint_boxes["U"] = (uln_x - 22, wrist_y - 16, 44, 34)

    # Carpal bones cluster
    carpal_y = wrist_y - 36
    carpal_offsets = [
        ("S", -30, 8),
        ("L", -6, 10),
        ("Tm", -44, -14),
        ("CMC-T", -54, -28),
        ("STT", -36, -6),
        ("SC", -18, -8),
        ("SR", -24, 18),
        ("CMC-I", -26, -24),
        ("CMC-M", -4, -26),
        ("CMC-R", 18, -24),
        ("CMC-S", 38, -20),
    ]
    for j_name, dx, dy in carpal_offsets:
        jx = cx + (dx if hand_side == "L" else -dx)
        jy = carpal_y + dy
        cv2.circle(canvas, (jx, jy), 13, 182.0, -1)
        cv2.circle(bone_mask, (jx, jy), 13, 3, -1)
        joint_boxes[j_name] = (jx - 20, jy - 20, 40, 40)

    # Five digits: T (Thumb), I (Index), M (Middle), R (Ring), S (Small)
    digit_specs = [
        ("T", -72, -32, 0.22),
        ("I", -40, -90, 0.34),
        ("M", 0, -105, 0.37),
        ("R", 38, -92, 0.34),
        ("S", 72, -70, 0.28),
    ]
    for code, dx_tip, dy_mcp, length_scale in digit_specs:
        dir_sign = 1 if hand_side == "L" else -1
        mcp_x = cx + int(dx_tip * 0.55 * dir_sign)
        mcp_y = carpal_y - 46 + (22 if code == "T" else 0)
        pip_x = cx + int(dx_tip * 0.82 * dir_sign)
        pip_y = mcp_y - int(height * length_scale * 0.48)
        tip_x = cx + int(dx_tip * 1.02 * dir_sign)
        tip_y = pip_y - int(height * length_scale * 0.42)

        # Metacarpal shaft
        base_x = cx + int(dx_tip * 0.30 * dir_sign)
        base_y = carpal_y - 12
        cv2.line(canvas, (base_x, base_y), (mcp_x, mcp_y), 192.0, 14)
        cv2.line(bone_mask, (base_x, base_y), (mcp_x, mcp_y), 4, 14)

        # Joint space gap at MCP
        gap = 2 if is_ra else 5
        cv2.circle(canvas, (mcp_x, mcp_y), 10, 198.0, -1)
        cv2.circle(canvas, (mcp_x, mcp_y - gap), 9, 45.0, -1)
        joint_boxes[f"MCP-{code}"] = (mcp_x - 24, mcp_y - 24, 48, 48)

        # Proximal Phalanx
        cv2.line(canvas, (mcp_x, mcp_y - gap - 2), (pip_x, pip_y), 188.0, 12)
        cv2.line(bone_mask, (mcp_x, mcp_y - gap - 2), (pip_x, pip_y), 5, 12)

        # PIP or Thumb IP joint
        pip_name = "IP" if code == "T" else f"PIP-{code}"
        joint_boxes[pip_name] = (pip_x - 22, pip_y - 22, 44, 44)

        # Distal / Middle Phalanx
        cv2.line(canvas, (pip_x, pip_y - gap), (tip_x, tip_y), 178.0, 10)
        cv2.line(bone_mask, (pip_x, pip_y - gap), (tip_x, tip_y), 5, 10)

        # Simulate bone erosions at MCP/PIP joints for RA cases
        if is_ra and code in ("I", "M", "R"):
            er_r = int(rng.integers(5, 10))
            er_x = mcp_x + int(rng.integers(-6, 7))
            er_y = mcp_y + int(rng.integers(-5, 6))
            cv2.circle(canvas, (er_x, er_y), er_r, 32.0, -1)
            er_cls = 2 if er_r < 8 else 3
            cv2.circle(erosion_mask, (er_x, er_y), er_r, er_cls, -1)

    # Smooth cortical bone transitions and add subtle quantum mottle noise
    canvas = cv2.GaussianBlur(canvas, (5, 5), 1.1)
    noise = rng.normal(0.0, 4.5, size=(height, width)).astype(np.float32)
    canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)
    return canvas, bone_mask, erosion_mask, joint_boxes


def generate_ram_h1200_fixtures(root_dir: str | Path = ".", seed: int = 42) -> Dict[str, Any]:
    """Populate `dataset/` with a complete RAM-H1200-compatible local fixture dataset."""
    root = Path(root_dir)
    raw_dir = root / "dataset" / "raw"
    processed_dir = root / "dataset" / "processed"
    splits_dir = root / "dataset" / "splits"

    for d in [
        raw_dir / "images",
        raw_dir / "annotations",
        raw_dir / "metadata",
        processed_dir / "images",
        processed_dir / "masks",
        processed_dir / "labels",
        splits_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    height, width = 512, 512

    # Strict patient-level splits matching RAM-H1200 structure
    split_specs = {
        "train": [
            ("JP_HMCRD_P0001_20210203_6791_L", "P0001", "HMCRD", 1, "F", 58, "L"),
            ("JP_HMCRD_P0001_20210203_6791_R", "P0001", "HMCRD", 1, "F", 58, "R"),
            ("JP_HMCRD_P0002_20210315_4102_L", "P0002", "HMCRD", 0, "M", 44, "L"),
            ("JP_HMCRD_P0002_20210315_4102_R", "P0002", "HMCRD", 0, "M", 44, "R"),
            ("JP_HU1_P0003_20201119_8821_L", "P0003", "HU1", 1, "F", 63, "L"),
            ("JP_HU1_P0003_20201119_8821_R", "P0003", "HU1", 1, "F", 63, "R"),
            ("JP_SARC_P0004_20190810_1209_L", "P0004", "SARC", 0, "F", 39, "L"),
            ("JP_SARC_P0004_20190810_1209_R", "P0004", "SARC", 0, "F", 39, "R"),
            ("JP_SCGH_P0005_20220114_5530_L", "P0005", "SCGH", 1, "M", 67, "L"),
            ("JP_SCGH_P0005_20220114_5530_R", "P0005", "SCGH", 1, "M", 67, "R"),
            ("JP_HU2_P0006_20210722_3318_L", "P0006", "HU2", 0, "M", 51, "L"),
            ("JP_HU2_P0006_20210722_3318_R", "P0006", "HU2", 0, "M", 51, "R"),
        ],
        "val": [
            ("JP_HMCRD_P0101_20210613_3770_L", "P0101", "HMCRD", 1, "F", 61, "L"),
            ("JP_HMCRD_P0101_20210613_3770_R", "P0101", "HMCRD", 1, "F", 61, "R"),
            ("JP_SARC_P0102_20200503_8272_L", "P0102", "SARC", 0, "F", 47, "L"),
            ("JP_SARC_P0102_20200503_8272_R", "P0102", "SARC", 0, "F", 47, "R"),
        ],
        "test": [
            ("JP_HMCRD_P0201_20210816_1549_L", "P0201", "HMCRD", 1, "F", 64, "L"),
            ("JP_HMCRD_P0201_20210816_1549_R", "P0201", "HMCRD", 1, "F", 64, "R"),
            ("JP_SCGH_P0202_20201129_0463_L", "P0202", "SCGH", 0, "M", 49, "L"),
            ("JP_SCGH_P0202_20201129_0463_R", "P0202", "SCGH", 0, "M", 49, "R"),
        ],
    }

    metadata_rows: List[Dict[str, Any]] = []
    split_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for split_name, items in split_specs.items():
        seg_split_dir = raw_dir / "Segmentation" / split_name
        be_split_dir = raw_dir / "SvdH_Scoring" / "SvdH_BE_Scoring" / split_name
        jsn_split_dir = raw_dir / "SvdH_Scoring" / "SvdH_JSN_Scoring" / split_name
        seg_split_dir.mkdir(parents=True, exist_ok=True)
        be_split_dir.mkdir(parents=True, exist_ok=True)
        jsn_split_dir.mkdir(parents=True, exist_ok=True)

        bone_coco = {"images": [], "annotations": [], "categories": BONE_CATEGORIES}
        be_seg_coco = {"images": [], "annotations": [], "categories": BE_SEG_CATEGORIES}
        be_det_coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": j} for i, j in enumerate(BE_JOINTS)],
        }
        jsn_det_coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i + 1, "name": j} for i, j in enumerate(JSN_JOINTS)],
        }
        be_scores_dict: Dict[str, Dict[str, int]] = {}
        jsn_scores_dict: Dict[str, Dict[str, int]] = {}

        ann_bone_id = 1
        ann_be_seg_id = 1
        ann_be_det_id = 1
        ann_jsn_det_id = 1

        for img_idx, (stem, patient_id, center, is_ra, sex, age, lr) in enumerate(items):
            bmp_name = f"{stem}.bmp"
            img_u8, bone_mask, erosion_mask, joint_boxes = _synthesize_hand_xray(
                height, width, bool(is_ra), lr, rng
            )

            # Save in RAM-H1200 Segmentation/<split>/ and flat raw/images/
            seg_img_path = seg_split_dir / bmp_name
            flat_img_path = raw_dir / "images" / bmp_name
            cv2.imwrite(str(seg_img_path), img_u8)
            cv2.imwrite(str(flat_img_path), img_u8)

            # Save combined semantic mask in processed/masks/
            combined_mask = np.where(erosion_mask > 0, erosion_mask + 5, bone_mask).astype(np.uint8)
            cv2.imwrite(str(processed_dir / "masks" / f"{stem}_mask.png"), combined_mask)
            cv2.imwrite(str(processed_dir / "masks" / f"{stem}_bone.png"), bone_mask)
            cv2.imwrite(str(processed_dir / "masks" / f"{stem}_erosion.png"), erosion_mask)

            img_entry = {
                "id": img_idx,
                "file_name": bmp_name,
                "width": width,
                "height": height,
            }
            bone_coco["images"].append(img_entry)
            be_seg_coco["images"].append(img_entry)
            be_det_coco["images"].append(img_entry)
            jsn_det_coco["images"].append(img_entry)

            # Bone RLE annotations
            for cat in BONE_CATEGORIES:
                cat_id = cat["id"]
                binary = (bone_mask == cat_id).astype(np.uint8)
                if binary.sum() > 0:
                    ys, xs = np.where(binary > 0)
                    bbox = [
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() - xs.min() + 1),
                        float(ys.max() - ys.min() + 1),
                    ]
                    bone_coco["annotations"].append(
                        {
                            "id": ann_bone_id,
                            "image_id": img_idx,
                            "category_id": cat_id,
                            "bbox": bbox,
                            "area": float(binary.sum()),
                            "iscrowd": 0,
                            "segmentation": encode_binary_mask_to_uncompressed_rle(binary),
                        }
                    )
                    ann_bone_id += 1

            # Erosion RLE annotations
            for cat in BE_SEG_CATEGORIES:
                cat_id = cat["id"]
                binary = (erosion_mask == cat_id).astype(np.uint8)
                if binary.sum() > 0:
                    ys, xs = np.where(binary > 0)
                    bbox = [
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() - xs.min() + 1),
                        float(ys.max() - ys.min() + 1),
                    ]
                    be_seg_coco["annotations"].append(
                        {
                            "id": ann_be_seg_id,
                            "image_id": img_idx,
                            "category_id": cat_id,
                            "bbox": bbox,
                            "area": float(binary.sum()),
                            "iscrowd": 0,
                            "segmentation": encode_binary_mask_to_uncompressed_rle(binary),
                        }
                    )
                    ann_be_seg_id += 1

            # ROI crops + BE & JSN joint detection + SvdH joint scores
            be_roi_dir = be_split_dir / stem
            jsn_roi_dir = jsn_split_dir / stem
            be_roi_dir.mkdir(parents=True, exist_ok=True)
            jsn_roi_dir.mkdir(parents=True, exist_ok=True)

            be_case_scores: Dict[str, int] = {}
            for j_idx, j_name in enumerate(BE_JOINTS):
                bx, by, bw, bh = joint_boxes.get(j_name, (width // 2 - 20, height // 2 - 20, 40, 40))
                x0 = max(0, min(width - 8, bx))
                y0 = max(0, min(height - 8, by))
                x1 = max(x0 + 8, min(width, bx + bw))
                y1 = max(y0 + 8, min(height, by + bh))
                roi = img_u8[y0:y1, x0:x1]
                cv2.imwrite(str(be_roi_dir / f"{j_name}.bmp"), roi)

                be_det_coco["annotations"].append(
                    {
                        "id": ann_be_det_id,
                        "image_id": img_idx,
                        "category_id": j_idx + 1,
                        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        "area": float((x1 - x0) * (y1 - y0)),
                        "iscrowd": 0,
                    }
                )
                ann_be_det_id += 1
                score_val = int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])) if is_ra else 0
                be_case_scores[f"BE_{j_name}"] = score_val

            jsn_case_scores: Dict[str, int] = {}
            for j_idx, j_name in enumerate(JSN_JOINTS):
                bx, by, bw, bh = joint_boxes.get(j_name, (width // 2 - 20, height // 2 - 20, 40, 40))
                x0 = max(0, min(width - 8, bx))
                y0 = max(0, min(height - 8, by))
                x1 = max(x0 + 8, min(width, bx + bw))
                y1 = max(y0 + 8, min(height, by + bh))
                roi = img_u8[y0:y1, x0:x1]
                cv2.imwrite(str(jsn_roi_dir / f"{j_name}.bmp"), roi)

                jsn_det_coco["annotations"].append(
                    {
                        "id": ann_jsn_det_id,
                        "image_id": img_idx,
                        "category_id": j_idx + 1,
                        "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                        "area": float((x1 - x0) * (y1 - y0)),
                        "iscrowd": 0,
                    }
                )
                ann_jsn_det_id += 1
                score_val = int(rng.choice([1, 2, 3], p=[0.45, 0.4, 0.15])) if is_ra else 0
                jsn_case_scores[f"JSN_{j_name}"] = score_val

            be_scores_dict[bmp_name] = be_case_scores
            jsn_scores_dict[bmp_name] = jsn_case_scores

            total_be = sum(be_case_scores.values())
            total_jsn = sum(jsn_case_scores.values())

            meta_row = {
                "Mapped Image Stem": stem,
                "StudyID": f"STUDY_{patient_id}_{stem.split('_')[3]}",
                "Normalized PatientID": f"JP_{center}_{patient_id}",
                "isRA": int(is_ra),
                "Sex": sex,
                "Age": int(age),
                "Center": center,
                "PixelSpacing": "0.15,0.15",
                "ImageSize": f"{height},{width}",
                "LR": lr,
                "Split": split_name,
                "Total_BE_Score": total_be,
                "Total_JSN_Score": total_jsn,
                "Total_SvdH_Score": total_be + total_jsn,
            }
            metadata_rows.append(meta_row)

            split_records[split_name].append(
                {
                    "image_stem": stem,
                    "file_name": bmp_name,
                    "image_path": str(seg_img_path.relative_to(root)).replace("\\", "/"),
                    "patient_id": f"JP_{center}_{patient_id}",
                    "center": center,
                    "is_ra": int(is_ra),
                    "sex": sex,
                    "age": int(age),
                    "hand_side": lr,
                    "total_be_score": total_be,
                    "total_jsn_score": total_jsn,
                    "total_svdh_score": total_be + total_jsn,
                    "split": split_name,
                }
            )

        # Write COCO & SvdH JSON files per split
        (seg_split_dir / "_annotations_bone_rle.coco.json").write_text(
            json.dumps(bone_coco, indent=2), encoding="utf-8"
        )
        (seg_split_dir / "_annotations_be_rle.coco.json").write_text(
            json.dumps(be_seg_coco, indent=2), encoding="utf-8"
        )
        (be_split_dir / "_annotations_be_joint_detection.coco.json").write_text(
            json.dumps(be_det_coco, indent=2), encoding="utf-8"
        )
        (be_split_dir / "_annotation_be_scores.json").write_text(
            json.dumps(be_scores_dict, indent=2), encoding="utf-8"
        )
        (jsn_split_dir / "_annotations_jsn_joint_detection.coco.json").write_text(
            json.dumps(jsn_det_coco, indent=2), encoding="utf-8"
        )
        (jsn_split_dir / "_annotation_jsn_scores.json").write_text(
            json.dumps(jsn_scores_dict, indent=2), encoding="utf-8"
        )

    # Write Metadata.xlsx and CSV splits
    meta_df = pd.DataFrame(metadata_rows)
    meta_xlsx_path = raw_dir / "Metadata.xlsx"
    meta_df.to_excel(meta_xlsx_path, index=False)
    meta_df.to_csv(raw_dir / "metadata" / "metadata.csv", index=False)

    pd.DataFrame(split_records["train"]).to_csv(splits_dir / "train.csv", index=False)
    pd.DataFrame(split_records["val"]).to_csv(splits_dir / "validation.csv", index=False)
    pd.DataFrame(split_records["test"]).to_csv(splits_dir / "test.csv", index=False)

    return {
        "total_images": len(metadata_rows),
        "train_count": len(split_records["train"]),
        "val_count": len(split_records["val"]),
        "test_count": len(split_records["test"]),
        "metadata_xlsx": str(meta_xlsx_path),
    }


if __name__ == "__main__":
    summary = generate_ram_h1200_fixtures(".")
    print("Generated RAM-H1200 local fixture dataset:", json.dumps(summary, indent=2))
