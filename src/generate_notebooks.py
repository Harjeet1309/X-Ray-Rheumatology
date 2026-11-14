"""Generate the 9 research Jupyter Notebooks specified in PRD Section 20."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

NOTEBOOK_SPECS: List[Tuple[str, str, str, str]] = [
    (
        "01_dataset_audit.ipynb",
        "01 — RAM-H1200 Dataset Audit & Patient-Level Split Verification",
        "Audits the official `TokyoTechMagicYang/RAM-H1200-v1` repository, inspects split counts, verifies zero patient overlap (`Train ∩ Val = 0`, `Train ∩ Test = 0`, `Val ∩ Test = 0`), and executes the 8-point validation pipeline.",
        "from src.data.download_ram_h1200 import fetch_remote_dataset_manifest\nfrom src.data.validation import run_dataset_validation\nimport json\n\nvalidation_report = run_dataset_validation('..')\nprint(json.dumps(validation_report, indent=2))",
    ),
    (
        "02_exploratory_data_analysis.ipynb",
        "02 — Exploratory Data Analysis (EDA) of Hand Radiographs & Annotations",
        "Analyzes dataset statistics, patient demographics (`Metadata.xlsx`), grayscale intensity histograms, contrast distributions, COCO RLE bone & erosion masks, and SvdH BE/JSN score distributions.",
        "import pandas as pd\nimport json\nfrom pathlib import Path\n\nmeta_df = pd.read_csv('../dataset/raw/metadata/metadata.csv')\nprint('Cohort Summary:')\nprint(meta_df[['isRA', 'Age', 'Total_BE_Score', 'Total_JSN_Score', 'Total_SvdH_Score']].describe())",
    ),
    (
        "03_preprocessing.ipynb",
        "03 — Radiograph Preprocessing, CLAHE & Clinically Constrained Augmentation",
        "Demonstrates aspect-ratio-preserving letterbox resize, CLAHE contrast enhancement, Gaussian/bilateral denoising, percentile normalization, and synchronized image+mask augmentations.",
        "from src.data.preprocessing import preprocess_xray\nfrom src.data.augmentation import RadiographAugmentor\nfrom pathlib import Path\n\nsample_bmp = next(Path('../dataset/raw/Segmentation/train').glob('*.bmp'))\nout = preprocess_xray(sample_bmp, target_size=(512, 512), use_clahe=True)\nprint('Preprocessed tensor shape:', out['tensor'].shape, 'Metadata:', out['meta'])",
    ),
    (
        "04_classification_training.ipynb",
        "04 — Model 1: Rheumatoid Arthritis Pattern Classification Training",
        "Trains and compares `DenseNet121`, `ResNet50`, and `EfficientNet-B0` for RA pattern classification (`isRA`) using transfer learning and clinical composite score selection.",
        "from src.classification.train import train_and_select_best_classifier\nimport json\n\nmeta = train_and_select_best_classifier('..', epochs=2, image_size=(96, 96))\nprint(json.dumps(meta, indent=2))",
    ),
    (
        "05_detection_training.ipynb",
        "05 — Model 2: SvdH Joint Localization & Abnormality Detection",
        "Trains the multi-joint localization detector using official COCO bounding-box annotations (`_annotations_be_joint_detection.coco.json` & `_annotations_jsn_joint_detection.coco.json`).",
        "from src.detection.train import train_detection_model\nimport json\n\nmeta = train_detection_model('..', epochs=5, image_size=(96, 96))\nprint(json.dumps(meta, indent=2))",
    ),
    (
        "06_segmentation_training.ipynb",
        "06 — Model 3: Hand Bone & Erosion Segmentation (U-Net & U-Net++)",
        "Decodes COCO RLE masks from `_annotations_bone_rle.coco.json` and `_annotations_be_rle.coco.json` and trains `U-Net` and `U-Net++` with combined Cross-Entropy + Multiclass Dice loss.",
        "from src.segmentation.train import train_segmentation_model\nimport json\n\nmeta = train_segmentation_model('..', epochs=3, image_size=(96, 96))\nprint(json.dumps(meta, indent=2))",
    ),
    (
        "07_severity_assessment.ipynb",
        "07 — Model 4: Sharp/van der Heijde (SvdH) Radiographic Severity Scoring",
        "Trains the multi-head ordinal severity network on 16 Bone Erosion (BE) joints (0–5) and 15 Joint Space Narrowing (JSN) joints (0–4) to estimate total hand SvdH damage (0–140).",
        "from src.severity.train import train_severity_model\nimport json\n\nmeta = train_severity_model('..', epochs=4, image_size=(96, 96))\nprint(json.dumps(meta, indent=2))",
    ),
    (
        "08_model_evaluation.ipynb",
        "08 — Independent Test Evaluation & Multi-Architecture Comparison",
        "Evaluates all 4 trained models on the held-out patient-separated test split using Sensitivity, Specificity, ROC-AUC, PR-AUC, mAP@50, Dice, IoU, MAE, RMSE, and Quadratic Weighted Kappa.",
        "from src.classification.evaluate import evaluate_classifier\nfrom src.detection.evaluate import evaluate_detector\nfrom src.segmentation.evaluate import evaluate_segmentation\nfrom src.severity.evaluate import evaluate_severity\nimport json\n\nprint('Classification:', json.dumps(evaluate_classifier('..')['metrics'], indent=2))\nprint('Detection:', json.dumps(evaluate_detector('..')['metrics'], indent=2))\nprint('Segmentation:', json.dumps(evaluate_segmentation('..')['metrics'], indent=2))\nprint('Severity:', json.dumps(evaluate_severity('..')['metrics'], indent=2))",
    ),
    (
        "09_explainable_ai.ipynb",
        "09 — Explainable AI: Grad-CAM Saliency & Multi-Modal Clinical Overlays",
        "Extracts Grad-CAM activation maps from the classification backbone and renders the 5 interactive diagnostic overlays (`original`, `gradcam`, `detection`, `segmentation`, `combined`).",
        "from src.classification.inference import ClassificationInferenceService\nfrom src.explainability.gradcam import GradCAMExtractor\nfrom src.data.preprocessing import preprocess_xray\nfrom pathlib import Path\n\nsample_bmp = next(Path('../dataset/raw/Segmentation/test').glob('*.bmp'))\ncls_service = ClassificationInferenceService('..')\ncam = GradCAMExtractor(cls_service.model, cls_service.model.get_target_conv_layer())\nprep = preprocess_xray(sample_bmp, target_size=cls_service.image_size, three_channel=True)\nheatmap = cam.generate_heatmap(prep['tensor'].unsqueeze(0), output_size=(512, 512))\nprint('Generated Grad-CAM heatmap shape:', heatmap.shape, 'min/max:', heatmap.min(), heatmap.max())",
    ),
]


def generate_notebooks(root_dir: str | Path = ".") -> None:
    nb_dir = Path(root_dir) / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)

    for fname, title, desc, code_block in NOTEBOOK_SPECS:
        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# {title}\n\n", f"{desc}\n"],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import sys\n",
                        "from pathlib import Path\n",
                        "sys.path.insert(0, str(Path('..').resolve()))\n\n",
                        code_block + "\n",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (nb_dir / fname).write_text(json.dumps(nb, indent=2), encoding="utf-8")


if __name__ == "__main__":
    generate_notebooks(".")
    print("Generated all 9 research notebooks in notebooks/")
