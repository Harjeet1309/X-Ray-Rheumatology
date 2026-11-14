"""SQLAlchemy ORM tables matching PRD Section 15 (Users, Analyses, Model_Results, Model_Registry, Reports)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    """Users table (PRD Section 15)."""

    __tablename__ = "users"

    user_id = Column(String(64), primary_key=True, default=_uuid)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(32), nullable=False, default="researcher")
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")


class Analysis(Base):
    """Analyses table (PRD Section 15)."""

    __tablename__ = "analyses"

    analysis_id = Column(String(64), primary_key=True, default=_uuid)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=True)
    filename = Column(String(255), nullable=False)
    image_reference = Column(String(512), nullable=False)
    upload_timestamp = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    analysis_status = Column(String(32), default="uploaded", nullable=False)
    completed_timestamp = Column(DateTime(timezone=True), nullable=True)
    quality_metadata = Column(JSON, nullable=True)
    summary_payload = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)

    user = relationship("User", back_populates="analyses")
    model_results = relationship(
        "ModelResult", back_populates="analysis", cascade="all, delete-orphan"
    )
    reports = relationship("Report", back_populates="analysis", cascade="all, delete-orphan")


class ModelResult(Base):
    """Model_Results table (PRD Section 15)."""

    __tablename__ = "model_results"

    result_id = Column(String(64), primary_key=True, default=_uuid)
    analysis_id = Column(String(64), ForeignKey("analyses.analysis_id"), nullable=False, index=True)
    model_name = Column(String(120), nullable=False)
    model_version = Column(String(32), nullable=False)
    prediction = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=True)
    output_reference = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    analysis = relationship("Analysis", back_populates="model_results")


class ModelRegistry(Base):
    """Model_Registry table (PRD Section 15)."""

    __tablename__ = "model_registry"

    model_id = Column(String(64), primary_key=True, default=_uuid)
    model_name = Column(String(120), nullable=False)
    task = Column(String(64), nullable=False)
    architecture = Column(String(120), nullable=False)
    version = Column(String(32), nullable=False)
    dataset_version = Column(String(64), nullable=False, default="RAM-H1200-v1")
    evaluation_metrics = Column(JSON, nullable=False)
    deployment_status = Column(String(32), nullable=False, default="active")
    registered_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)


class Report(Base):
    """Reports table (PRD Section 15)."""

    __tablename__ = "reports"

    report_id = Column(String(64), primary_key=True, default=_uuid)
    analysis_id = Column(String(64), ForeignKey("analyses.analysis_id"), nullable=False, index=True)
    report_reference = Column(String(512), nullable=False)
    generated_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    analysis = relationship("Analysis", back_populates="reports")
