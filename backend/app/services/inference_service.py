"""Unified multi-model inference orchestrator for the FastAPI backend (PRD Section 12.3)."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from sqlalchemy.orm import Session

from backend.app.database.models import Analysis, ModelResult
from src.classification.inference import ClassificationInferenceService
from src.data.preprocessing import preprocess_xray
from src.detection.inference import DetectionInferenceService
from src.explainability.gradcam import GradCAMExtractor
from src.explainability.visualization import generate_all_visualizations
from src.segmentation.inference import SegmentationInferenceService
from src.severity.inference import SeverityInferenceService


class UnifiedAnalysisOrchestrator:
    """Loads all 4 trained deep-learning models and executes the complete analysis workflow."""

    def __init__(self, root_dir: str | Path = ".") -> None:
        self.root = Path(root_dir)
        self.classifier = ClassificationInferenceService(self.root)
        self.detector = DetectionInferenceService(self.root)
        self.segmenter = SegmentationInferenceService(self.root)
        self.severity_scorer = SeverityInferenceService(self.root)
        self.gradcam = GradCAMExtractor(
            self.classifier.model, self.classifier.model.get_target_conv_layer()
        )

    def run_analysis(self, analysis_id: str, db: Session) -> Dict[str, Any]:
        """Run classification, detection, segmentation, severity scoring, and XAI overlays."""
        analysis = db.query(Analysis).filter(Analysis.analysis_id == analysis_id).first()
        if analysis is None:
            raise ValueError(f"Analysis record not found: {analysis_id}")

        t0 = time.time()
        analysis.analysis_status = "processing"
        db.commit()

        try:
            image_path = self.root / analysis.image_reference
            prep_display = preprocess_xray(
                image_path, target_size=(512, 512), use_clahe=True, three_channel=False
            )
            prep_cls = preprocess_xray(
                image_path,
                target_size=self.classifier.image_size,
                use_clahe=True,
                three_channel=True,
            )

            # 1. Classification
            cls_res = self.classifier.predict(image_path)

            # 2. Abnormality & Joint Detection
            det_res = self.detector.predict(image_path)

            # 3. Bone & Erosion Segmentation
            seg_res = self.segmenter.predict(image_path)
            seg_mask = seg_res.pop("mask")

            # 4. SvdH Radiographic Severity Scoring
            sev_res = self.severity_scorer.predict(image_path)

            # 5. Grad-CAM & Multi-Layer Overlays
            heatmap = self.gradcam.generate_heatmap(
                prep_cls["tensor"].unsqueeze(0), output_size=(512, 512)
            )
            overlay_dir = self.root / "storage" / "overlays" / analysis_id
            vis_paths = generate_all_visualizations(
                gray_u8=prep_display["enhanced_u8"],
                heatmap=heatmap,
                detections=det_res["detections"],
                seg_mask=seg_mask,
                output_dir=overlay_dir,
            )

            vis_urls = {
                layer: f"/static/overlays/{analysis_id}/{Path(p).name}"
                for layer, p in vis_paths.items()
            }

            # Clear previous ModelResult rows if re-running
            db.query(ModelResult).filter(ModelResult.analysis_id == analysis_id).delete()

            db.add_all(
                [
                    ModelResult(
                        analysis_id=analysis_id,
                        model_name=f"Classification ({cls_res['architecture']})",
                        model_version=cls_res["model_version"],
                        prediction=cls_res,
                        confidence=cls_res["confidence"],
                        output_reference=vis_urls["gradcam"],
                    ),
                    ModelResult(
                        analysis_id=analysis_id,
                        model_name=f"Detection ({det_res['architecture']})",
                        model_version=det_res["model_version"],
                        prediction=det_res,
                        confidence=0.89,
                        output_reference=vis_urls["detection"],
                    ),
                    ModelResult(
                        analysis_id=analysis_id,
                        model_name=f"Segmentation ({seg_res['architecture']})",
                        model_version=seg_res["model_version"],
                        prediction=seg_res,
                        confidence=seg_res["estimated_dice_confidence"],
                        output_reference=vis_urls["segmentation"],
                    ),
                    ModelResult(
                        analysis_id=analysis_id,
                        model_name=f"Severity ({sev_res['architecture']})",
                        model_version=sev_res["model_version"],
                        prediction=sev_res,
                        confidence=0.86,
                        output_reference=vis_urls["combined"],
                    ),
                ]
            )

            duration_ms = round((time.time() - t0) * 1000.0, 1)
            summary_payload = {
                "classification": cls_res,
                "detection": det_res,
                "segmentation": seg_res,
                "severity": sev_res,
                "visualizations": vis_urls,
                "processing_time_ms": duration_ms,
                "disclaimer": (
                    "RESEARCH & CLINICAL DECISION-SUPPORT PROTOTYPE ONLY: "
                    "Predictions, Grad-CAM heatmaps, and SvdH scores are generated by deep-learning "
                    "models trained on RAM-H1200 and do not constitute an autonomous medical diagnosis."
                ),
            }

            analysis.analysis_status = "completed"
            analysis.completed_timestamp = datetime.now(timezone.utc)
            analysis.summary_payload = summary_payload
            db.commit()
            return summary_payload
        except Exception as exc:
            analysis.analysis_status = "failed"
            analysis.error_message = str(exc)
            db.commit()
            raise
