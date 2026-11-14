"""Authentication routes (/api/auth/register, /api/auth/login, /api/auth/me)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.models import User
from backend.app.database.session import get_db
from backend.app.middleware.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from backend.app.schemas.api_schemas import (
    AuthTokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
def register_user(payload: UserRegisterRequest, db: Session = Depends(get_db)) -> AuthTokenResponse:
    """Register a new researcher/medical user account."""
    email_norm = payload.email.lower().strip()
    existing = db.query(User).filter(User.email == email_norm).first()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email is already registered.",
        )

    user = User(
        name=payload.name.strip(),
        email=email_norm,
        password_hash=hash_password(payload.password),
        role="researcher",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.user_id, "email": user.email, "role": user.role})
    return AuthTokenResponse(
        access_token=token,
        user=UserResponse(
            user_id=user.user_id,
            name=user.name,
            email=user.email,
            role=user.role,
            created_at=user.created_at,
        ),
    )


@router.post("/login", response_model=AuthTokenResponse)
def login_user(payload: UserLoginRequest, db: Session = Depends(get_db)) -> AuthTokenResponse:
    """Authenticate an existing user and return a signed JWT session token."""
    email_norm = payload.email.lower().strip()
    user = db.query(User).filter(User.email == email_norm).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    token = create_access_token({"sub": user.user_id, "email": user.email, "role": user.role})
    return AuthTokenResponse(
        access_token=token,
        user=UserResponse(
            user_id=user.user_id,
            name=user.name,
            email=user.email,
            role=user.role,
            created_at=user.created_at,
        ),
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return profile of the currently authenticated user."""
    return UserResponse(
        user_id=current_user.user_id,
        name=current_user.name,
        email=current_user.email,
        role=current_user.role,
        created_at=current_user.created_at,
    )
