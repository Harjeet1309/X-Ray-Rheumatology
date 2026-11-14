"""SQLAlchemy session management and automatic Model Registry seeding."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.database.models import Base, ModelRegistry

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./xray_rheumatology.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(root_dir: str | Path = ".") -> None:
    """Create tables and synchronize trained model metadata into Model_Registry."""
    Base.metadata.create_all(bind=engine)
    root = Path(root_dir)
    db = SessionLocal()
    try:
        for task in ("classification", "detection", "segmentation", "severity"):
            meta_file = root / "models" / task / "metadata.json"
            test_file = root / "models" / task / "test_evaluation.json"
            if not meta_file.exists():
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            metrics = meta.get("evaluation_metrics", {})
            if test_file.exists():
                test_data = json.loads(test_file.read_text(encoding="utf-8"))
                metrics = test_data.get("metrics", metrics)

            existing = (
                db.query(ModelRegistry)
                .filter(
                    ModelRegistry.task == task,
                    ModelRegistry.version == meta.get("version", "1.0.0"),
                )
                .first()
            )
            if existing is None:
                db.add(
                    ModelRegistry(
                        model_name=meta.get("model_name", task.title()),
                        task=task,
                        architecture=meta.get("architecture", "unknown"),
                        version=meta.get("version", "1.0.0"),
                        dataset_version=meta.get("dataset_version", "RAM-H1200-v1"),
                        evaluation_metrics=metrics,
                        deployment_status="production",
                    )
                )
            else:
                existing.architecture = meta.get("architecture", existing.architecture)
                existing.evaluation_metrics = metrics
        db.commit()
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
