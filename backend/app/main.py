"""FastAPI application entry point for the AI-Powered Rheumatological X-ray Analysis Platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.database.session import init_db
from backend.app.routes import analysis, auth, models


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize storage directories and synchronize database & model registry on startup."""
    for d in ["storage/uploads", "storage/overlays", "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    init_db(".")
    yield


app = FastAPI(
    title="AI-Powered Rheumatological X-ray Analysis Platform API",
    description=(
        "End-to-End Deep Learning Backend for Hand X-Ray Rheumatoid Arthritis Classification, "
        "SvdH Joint Localization, Bone & Erosion Segmentation, Severity Scoring, and Grad-CAM Explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path("storage").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="storage"), name="static")

app.include_router(auth.router)
app.include_router(analysis.router)
app.include_router(models.router)


@app.get("/api/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Backend and ML model availability health check (PRD Section 12.2)."""
    model_files = {
        "classification": Path("models/classification/best_model.pth").exists(),
        "detection": Path("models/detection/best.pt").exists(),
        "segmentation": Path("models/segmentation/best_model.pth").exists(),
        "severity": Path("models/severity/best_model.pth").exists(),
    }
    return {
        "status": "healthy",
        "service": "rheumatology-xray-ai-backend",
        "version": "1.0.0",
        "models_ready": all(model_files.values()),
        "model_artifacts": model_files,
    }
