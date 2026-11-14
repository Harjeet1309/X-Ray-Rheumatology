"""Password hashing and JWT session token authentication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.app.database.models import User
from backend.app.database.session import get_db

JWT_SECRET = os.environ.get(
    "JWT_SECRET_KEY", "change-this-in-production-rheumatology-xray-secret-key-2026"
)
TOKEN_TTL_SECONDS = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")) * 60

bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256 with random salt."""
    salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120000)
    return f"pbkdf2_sha256${salt}${dk.hex()}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """Verify a password against its stored PBKDF2-HMAC-SHA256 hash."""
    try:
        scheme, salt, hex_digest = stored_hash.split("$", 2)
        if scheme != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), 120000
        )
        return hmac.compare_digest(dk.hex(), hex_digest)
    except Exception:
        return False


def create_access_token(payload: Dict[str, Any], expires_in: int = TOKEN_TTL_SECONDS) -> str:
    """Create a signed HMAC-SHA256 JWT token."""
    header = {"alg": "HS256", "typ": "JWT"}
    body = {**payload, "exp": int(time.time()) + expires_in}
    h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode("utf-8")).rstrip(b"=").decode("ascii")
    b_b64 = base64.urlsafe_b64encode(json.dumps(body).encode("utf-8")).rstrip(b"=").decode("ascii")
    signing_input = f"{h_b64}.{b_b64}".encode("ascii")
    sig = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    s_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode("ascii")
    return f"{h_b64}.{b_b64}.{s_b64}"


def decode_access_token(token: str) -> Dict[str, Any]:
    """Verify and decode a signed HMAC-SHA256 JWT token."""
    try:
        h_b64, b_b64, s_b64 = token.split(".")
        signing_input = f"{h_b64}.{b_b64}".encode("ascii")
        expected_sig = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
        expected_b64 = base64.urlsafe_b64encode(expected_sig).rstrip(b"=").decode("ascii")
        if not hmac.compare_digest(s_b64, expected_b64):
            raise ValueError("Invalid token signature")
        padded = b_b64 + "=" * (-len(b_b64) % 4)
        body = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        if int(body.get("exp", 0)) < int(time.time()):
            raise ValueError("Token expired")
        return body
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {exc}",
        ) from exc


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Return authenticated User if Bearer token is provided, or None for guest research mode."""
    if credentials is None or not credentials.credentials:
        return None
    data = decode_access_token(credentials.credentials)
    user_id = data.get("sub")
    if not user_id:
        return None
    return db.query(User).filter(User.user_id == user_id).first()


def get_current_user(
    user: Optional[User] = Depends(get_optional_user),
) -> User:
    """Require an authenticated user."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user
