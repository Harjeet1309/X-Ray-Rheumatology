"""Pydantic request and response schemas for the Rheumatological X-ray Analysis API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class UserRegisterRequest(BaseSchema):
    name: str = Field(..., min_length=2, max_length=120)
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(..., min_length=6, max_length=128)


class UserLoginRequest(BaseSchema):
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str


class UserResponse(BaseSchema):
    user_id: str
    name: str
    email: str
    role: str
    created_at: datetime


class AuthTokenResponse(BaseSchema):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class AnalysisUploadResponse(BaseSchema):
    analysis_id: str
    filename: str
    analysis_status: str
    upload_timestamp: datetime
    quality_metadata: Dict[str, Any]
    preview_url: str


class AnalysisRunResponse(BaseSchema):
    analysis_id: str
    analysis_status: str
    completed_timestamp: Optional[datetime] = None
    summary: Dict[str, Any]


class AnalysisDetailResponse(BaseSchema):
    analysis_id: str
    filename: str
    upload_timestamp: datetime
    completed_timestamp: Optional[datetime] = None
    analysis_status: str
    quality_metadata: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    model_results: List[Dict[str, Any]] = []
    report_available: bool = False


class ModelInfoResponse(BaseSchema):
    model_id: str
    model_name: str
    task: str
    architecture: str
    version: str
    dataset_version: str
    evaluation_metrics: Dict[str, Any]
    deployment_status: str
