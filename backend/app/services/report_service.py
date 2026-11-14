"""Clinical Research PDF Report Generator using ReportLab (PRD Section 13.2 Page 8)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy.orm import Session

from backend.app.database.models import Analysis, Report


def generate_analysis_pdf_report(
    analysis: Analysis,
    db: Session,
    root_dir: str | Path = ".",
) -> Path:
    """Generate a structured PDF report for a completed X-ray analysis and record it in DB."""
    root = Path(root_dir)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pdf_filename = f"rheumatology_report_{analysis.analysis_id[:8]}.pdf"
    pdf_path = reports_dir / pdf_filename

    summary: Dict[str, Any] = analysis.summary_payload or {}
    cls_data = summary.get("classification", {})
    det_data = summary.get("detection", {})
    seg_data = summary.get("segmentation", {})
    sev_data = summary.get("severity", {})
    qual_data = analysis.quality_metadata or {}

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Rheumatological X-Ray AI Analysis Report - {analysis.analysis_id[:8]}",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#0D9488"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodySmall",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerBox",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#7F1D1D"),
        backColor=colors.HexColor("#FEF2F2"),
        borderPadding=6,
    )

    story = []
    story.append(
        Paragraph("AI-Powered Rheumatological Hand X-Ray Analysis Report", title_style)
    )
    story.append(
        Paragraph(
            f"<b>Analysis ID:</b> {analysis.analysis_id} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Image File:</b> {analysis.filename} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            subtitle_style,
        )
    )

    # 1. Summary Metadata & Classification Table
    story.append(Paragraph("1. Radiographic Classification & Model Versions", section_style))
    meta_rows = [
        ["Metric / Property", "Observation", "Model Version / Architecture"],
        [
            "RA Classification Finding",
            str(cls_data.get("predicted_class", "N/A")),
            f"{cls_data.get('architecture', 'densenet121')} (v{cls_data.get('model_version', '1.0.0')})",
        ],
        [
            "Calibrated RA Probability",
            f"{float(cls_data.get('ra_probability', 0.0)) * 100:.1f}% (Confidence: {float(cls_data.get('confidence', 0.0)) * 100:.1f}%)",
            "Temperature-Calibrated Softmax",
        ],
        [
            "Joint & Erosion Localization",
            f"{det_data.get('total_joints_localized', 16)} SvdH joints localized; "
            f"{det_data.get('abnormal_regions_flagged', 0)} flagged",
            f"{det_data.get('architecture', 'yolov8')} (v{det_data.get('model_version', '1.0.0')})",
        ],
        [
            "Bone & Erosion Segmentation",
            f"Bone Coverage: {seg_data.get('bone_coverage_percent', 0)}% | "
            f"Erosion Ratio: {seg_data.get('erosion_area_percent', 0)}%",
            f"{seg_data.get('architecture', 'unet')} (v{seg_data.get('model_version', '1.0.0')})",
        ],
        [
            "Input Radiograph Quality",
            f"{qual_data.get('width', 512)}x{qual_data.get('height', 512)}px | "
            f"Contrast SD: {qual_data.get('std_intensity', 'N/A')}",
            "CLAHE + Aspect-Preserving Resize",
        ],
    ]
    t1 = Table(meta_rows, colWidths=[48 * mm, 76 * mm, 52 * mm])
    t1.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(t1)

    # 2. SvdH Severity Assessment Breakdown
    story.append(
        Paragraph("2. Sharp/van der Heijde (SvdH) Radiographic Severity Estimate", section_style)
    )
    story.append(
        Paragraph(
            f"<b>Estimated Hand Bone Erosion (BE) Score:</b> {sev_data.get('total_be_score', 0)} / 80 &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Joint Space Narrowing (JSN) Score:</b> {sev_data.get('total_jsn_score', 0)} / 60 &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"<b>Combined Hand SvdH Score:</b> {sev_data.get('total_svdh_score', 0)} / 140 "
            f"({sev_data.get('severity_category', 'N/A')})",
            body_style,
        )
    )
    story.append(Spacer(1, 3 * mm))

    be_joints = sev_data.get("be_joint_scores", {})
    jsn_joints = sev_data.get("jsn_joint_scores", {})
    joint_rows = [["Joint ROI", "BE Score (0-5)", "JSN Score (0-4)", "Joint ROI", "BE Score (0-5)", "JSN Score (0-4)"]]
    all_keys = list(be_joints.keys())
    half = (len(all_keys) + 1) // 2
    for i in range(half):
        k1 = all_keys[i]
        k2 = all_keys[i + half] if (i + half) < len(all_keys) else ""
        joint_rows.append(
            [
                k1,
                str(be_joints.get(k1, "-")),
                str(jsn_joints.get(k1, "-")),
                k2,
                str(be_joints.get(k2, "-")) if k2 else "-",
                str(jsn_joints.get(k2, "-")) if k2 else "-",
            ]
        )

    t2 = Table(joint_rows, colWidths=[28 * mm, 30 * mm, 30 * mm, 28 * mm, 30 * mm, 30 * mm])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
            ]
        )
    )
    story.append(t2)

    # 3. Multi-Modal Visual Evidence
    story.append(Paragraph("3. Explainable AI & Multi-Modal Visual Overlays", section_style))
    overlay_dir = root / "storage" / "overlays" / analysis.analysis_id
    if overlay_dir.exists():
        img_cells = []
        captions = []
        for fname, label in [
            ("original.png", "Original Enhanced X-Ray"),
            ("gradcam.png", "Grad-CAM Saliency Heatmap"),
            ("detection.png", "SvdH Joint Localization"),
            ("segmentation.png", "Bone & Erosion Segmentation"),
        ]:
            fpath = overlay_dir / fname
            if fpath.exists():
                img_cells.append(RLImage(str(fpath), width=41 * mm, height=41 * mm))
                captions.append(label)
        if img_cells:
            vis_table = Table([img_cells, captions], colWidths=[44 * mm] * len(img_cells))
            vis_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTSIZE", (0, 1), (-1, 1), 7.5),
                        ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#334155")),
                    ]
                )
            )
            story.append(vis_table)

    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(
            "<b>CLINICAL RESEARCH & UNCERTAINTY DISCLAIMER:</b> This report contains automated observations "
            "generated by research deep-learning models trained on the RAM-H1200 dataset. Grad-CAM heatmaps, "
            "segmentation overlays, and SvdH score estimates indicate regions of model attention and do not "
            "constitute confirmed clinical findings or a medical diagnosis. All radiographic interpretations "
            "must be verified by a qualified radiologist or rheumatologist.",
            disclaimer_style,
        )
    )

    doc.build(story)

    # Save Report record in database
    existing_rep = db.query(Report).filter(Report.analysis_id == analysis.analysis_id).first()
    rel_ref = str(pdf_path.relative_to(root)).replace("\\", "/")
    if existing_rep is None:
        db.add(Report(analysis_id=analysis.analysis_id, report_reference=rel_ref))
    else:
        existing_rep.report_reference = rel_ref
        existing_rep.generated_at = datetime.now(timezone.utc)
    db.commit()

    return pdf_path
