from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import AUTH_SECRET, TOKEN_TTL_MINUTES


@dataclass
class AuthUser:
    username: str
    role: str
    merchant_id: str | None = None


_DEMO_USERS: dict[str, dict[str, str | None]] = {
    "admin": {
        "password": "admin123",
        "role": "analyst",
        "merchant_id": None,
    },
    "ops_lead": {
        "password": "ops123",
        "role": "analyst",
        "merchant_id": None,
    },
    "electro_mgr": {
        "password": "electro123",
        "role": "merchant_admin",
        "merchant_id": "MRT-ELECTRO-01",
    },
    "grocery_mgr": {
        "password": "grocery123",
        "role": "merchant_admin",
        "merchant_id": "MRT-GROCERY-07",
    },
}

security = HTTPBearer(auto_error=False)


def authenticate_user(username: str, password: str) -> AuthUser | None:
    user = _DEMO_USERS.get(username)
    if not user:
        return None

    if not hmac.compare_digest(str(user["password"]), password):
        return None

    return AuthUser(
        username=username,
        role=str(user["role"]),
        merchant_id=_normalize_optional(user["merchant_id"]),
    )


def create_access_token(user: AuthUser) -> tuple[str, int]:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MINUTES)
    payload = {
        "sub": user.username,
        "role": user.role,
        "merchant_id": user.merchant_id,
        "exp": int(expires_at.timestamp()),
    }

    token = _encode_signed_payload(payload)
    return token, TOKEN_TTL_MINUTES * 60


def decode_access_token(token: str) -> AuthUser:
    payload = _decode_signed_payload(token)

    exp = int(payload.get("exp", 0))
    if exp <= int(datetime.now(timezone.utc).timestamp()):
        raise _unauthorized("Token expired")

    username = str(payload.get("sub", ""))
    role = str(payload.get("role", ""))
    merchant_id = _normalize_optional(payload.get("merchant_id"))

    if not username or role not in {"analyst", "merchant_admin"}:
        raise _unauthorized("Invalid token")

    return AuthUser(username=username, role=role, merchant_id=merchant_id)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> AuthUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized("Authentication required")

    return decode_access_token(credentials.credentials)


def require_analyst(user: AuthUser) -> None:
    if user.role != "analyst":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Analyst access required",
        )


def ensure_merchant_scope(user: AuthUser, merchant_id: str | None) -> str | None:
    if user.role != "merchant_admin":
        return merchant_id

    if user.merchant_id is None:
        raise HTTPException(status_code=403, detail="Merchant scope is missing for user")

    if merchant_id is None or merchant_id == user.merchant_id:
        return user.merchant_id

    raise HTTPException(status_code=403, detail="Not allowed for this merchant")


def user_profile_dict(user: AuthUser) -> dict[str, Any]:
    return {
        "username": user.username,
        "role": user.role,
        "merchant_id": user.merchant_id,
    }


def _encode_signed_payload(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    payload_part = _b64url_encode(serialized)
    signature = _sign(payload_part)
    return f"{payload_part}.{signature}"


def _decode_signed_payload(token: str) -> dict[str, Any]:
    try:
        payload_part, signature = token.split(".", 1)
    except ValueError as exc:
        raise _unauthorized("Malformed token") from exc

    expected_signature = _sign(payload_part)
    if not hmac.compare_digest(signature, expected_signature):
        raise _unauthorized("Invalid token signature")

    try:
        raw = _b64url_decode(payload_part)
        payload = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise _unauthorized("Invalid token payload") from exc

    if not isinstance(payload, dict):
        raise _unauthorized("Invalid token payload")

    return payload


def _sign(message: str) -> str:
    digest = hmac.new(AUTH_SECRET.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).digest()
    return _b64url_encode(digest)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _normalize_optional(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=message,
        headers={"WWW-Authenticate": "Bearer"},
    )
