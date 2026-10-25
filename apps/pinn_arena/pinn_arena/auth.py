"""
GitHub OAuth 2.0 sign-in for the PINNeAPPle Arena (optional: experiments run without it, as a guest).

Flow:
  GET  /api/auth/github/login      -> redirect to GitHub authorization page
  GET  /api/auth/github/callback   -> exchange code, upsert user, issue JWT
  GET  /api/auth/me                -> return current user (requires JWT)
  POST /api/auth/logout            -> client-side only; endpoint for completeness

The JWT travels in the ``X-Arena-Token`` header (``Authorization: Bearer`` is also accepted), so it does not
collide with the HTTP Basic login of ``PINN_ARENA_USER`` / ``PINN_ARENA_PASSWORD`` (appkit) on a private
deployment. GitHub's access token is used once, to read the profile, and is not stored.

Environment (no defaults that are secrets):
  GITHUB_CLIENT_ID, GITHUB_CLIENT_SECRET   the OAuth app (https://github.com/settings/developers); unset: sign-in off
  GITHUB_CALLBACK_URL                      default ``<this server>/api/auth/github/callback``
  ARENA_JWT_SECRET                         HS256 key of the session tokens; unset: a random key per process
                                           (every restart signs everyone out)
  ARENA_FRONTEND_URL                       where to send the browser after sign-in; default ``/`` (this server)
"""
# ruff: noqa: B008  (FastAPI Depends/Query/File defaults)
from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from jwt import PyJWTError
from sqlalchemy.orm import Session

from pinn_arena.db import get_db
from pinn_arena.orm import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# Configuration (from environment variables with sane defaults)
# ---------------------------------------------------------------------------

GITHUB_CLIENT_ID: str = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET: str = os.environ.get("GITHUB_CLIENT_SECRET", "")
GITHUB_CALLBACK_URL: str = os.environ.get("GITHUB_CALLBACK_URL", "")
_jwt_secret_from_env = os.environ.get("ARENA_JWT_SECRET", "")
if _jwt_secret_from_env:
    ARENA_JWT_SECRET: str = _jwt_secret_from_env
else:
    ARENA_JWT_SECRET = secrets.token_hex(32)
    logger.warning(
        "ARENA_JWT_SECRET not set — using a random secret. "
        "All existing JWTs will be invalidated on restart. "
        "Set ARENA_JWT_SECRET for persistent sessions."
    )
ARENA_FRONTEND_URL: str = os.environ.get("ARENA_FRONTEND_URL", "").rstrip("/")

ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30

# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def _create_access_token(github_id: int, username: str) -> str:
    expire = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(github_id),
        "username": username,
        "exp": expire,
    }
    return jwt.encode(payload, ARENA_JWT_SECRET, algorithm=ALGORITHM)


def _decode_token(token: str) -> dict:
    """Decode and validate a JWT. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, ARENA_JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# ---------------------------------------------------------------------------
# FastAPI security scheme (Bearer token)
# ---------------------------------------------------------------------------

def _token(request: Request) -> str | None:
    """The session JWT: ``X-Arena-Token``, else ``Authorization: Bearer``."""
    tok = request.headers.get("x-arena-token")
    if tok:
        return tok
    auth = request.headers.get("authorization", "")
    return auth[7:] if auth.lower().startswith("bearer ") else None


def get_current_user_optional(request: Request, db: Session = Depends(get_db)) -> User | None:
    """Return the authenticated user or None (auth is optional)."""
    token = _token(request)
    if token is None:
        return None
    try:
        payload = _decode_token(token)
        github_id = int(payload["sub"])
    except (HTTPException, KeyError, ValueError):
        return None

    user = db.query(User).filter(User.github_id == github_id).first()
    return user


def get_current_user_required(request: Request, db: Session = Depends(get_db)) -> User:
    """Return the authenticated user or raise 401."""
    user = get_current_user_optional(request, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ---------------------------------------------------------------------------
# OAuth endpoints
# ---------------------------------------------------------------------------


def _callback_url(request: Request) -> str:
    return GITHUB_CALLBACK_URL or str(request.base_url).rstrip("/") + "/api/auth/github/callback"


@router.get("/github/login", summary="Redirect to GitHub OAuth")
def github_login(request: Request):
    """
    Redirect the browser to the GitHub OAuth authorization page.

    Generates a random `state` token and stores it in an HTTP-only cookie to
    prevent CSRF attacks on the OAuth callback.
    """
    if not GITHUB_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth is not configured (GITHUB_CLIENT_ID missing).",
        )

    state = secrets.token_urlsafe(16)
    params = urlencode({"client_id": GITHUB_CLIENT_ID, "redirect_uri": _callback_url(request),
                        "scope": "read:user user:email", "state": state})
    response = RedirectResponse(
        url=f"https://github.com/login/oauth/authorize?{params}",
        status_code=302,
    )
    # httponly=True keeps the cookie out of JavaScript; secure=True enforces HTTPS in prod
    response.set_cookie(
        "oauth_state",
        state,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https" or ARENA_FRONTEND_URL.startswith("https://"),
        max_age=300,   # 5 minutes — enough for the OAuth round-trip
        path="/api/auth",
    )
    return response


@router.get("/github/callback", summary="GitHub OAuth callback")
async def github_callback(
    request: Request,
    code: str,
    state: str = "",
    db: Session = Depends(get_db),
):
    """
    Handle the GitHub OAuth callback:
    1. Verify CSRF state parameter against the cookie set in /github/login.
    2. Exchange `code` for a GitHub access token.
    3. Fetch the GitHub user profile.
    4. Upsert the user in the database.
    5. Issue a JWT and redirect to the frontend via URL fragment (#token=...).
    """
    # CSRF check — state must match the cookie set during login
    expected_state = request.cookies.get("oauth_state", "")
    if not expected_state or not secrets.compare_digest(state, expected_state):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid OAuth state parameter. Please try logging in again.",
        )

    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GitHub OAuth is not configured on the server.",
        )

    async with httpx.AsyncClient() as client:
        # Step 1: Exchange code for access token
        token_resp = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": _callback_url(request),
            },
            headers={"Accept": "application/json"},
            timeout=15,
        )

    if token_resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to exchange code with GitHub.",
        )

    token_data = token_resp.json()
    gh_access_token = token_data.get("access_token")
    if not gh_access_token:
        error_desc = token_data.get("error_description", "Unknown error")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"GitHub returned no access token: {error_desc}",
        )

    # Step 2: Fetch user profile
    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            "https://api.github.com/user",
            headers={
                "Authorization": f"token {gh_access_token}",
                "Accept": "application/vnd.github.v3+json",
            },
            timeout=15,
        )

    if user_resp.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch user profile from GitHub.",
        )

    gh_user = user_resp.json()
    github_id: int = int(gh_user["id"])
    username: str = gh_user["login"]

    # Step 3: Upsert user in DB
    user = db.query(User).filter(User.github_id == github_id).first()
    if user is None:
        user = User(
            github_id=github_id,
            username=username,
            name=gh_user.get("name"),
            email=gh_user.get("email"),
            avatar_url=gh_user.get("avatar_url"),
        )
        db.add(user)
        logger.info("New user registered: %s (github_id=%d)", username, github_id)
    else:
        user.username = username
        user.name = gh_user.get("name")
        user.email = gh_user.get("email")
        user.avatar_url = gh_user.get("avatar_url")
        user.last_login = datetime.utcnow()
        logger.info("Returning user login: %s (github_id=%d)", username, github_id)

    db.commit()
    db.refresh(user)

    # Step 5: Issue JWT and redirect to frontend
    # Use URL fragment (#) so the token is never sent to the server in logs or
    # Referer headers — it lives only in the browser until the JS reads it.
    jwt_token = _create_access_token(github_id, username)
    redirect_url = f"{ARENA_FRONTEND_URL or ''}/#token={jwt_token}"
    response = RedirectResponse(url=redirect_url, status_code=302)
    response.delete_cookie("oauth_state", path="/api/auth")
    return response


@router.get("/me", summary="Get current authenticated user")
def get_me(current_user: User = Depends(get_current_user_required)):
    """Return the profile of the currently authenticated user."""
    return {
        "github_id": current_user.github_id,
        "username": current_user.username,
        "name": current_user.name,
        "email": current_user.email,
        "avatar_url": current_user.avatar_url,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
    }


@router.post("/logout", summary="Logout (client-side token removal)")
def logout():
    """
    Logout endpoint.

    JWT tokens are stateless — actual logout is handled client-side by
    removing the token from localStorage.  This endpoint exists for
    completeness and future server-side token revocation.
    """
    return {"message": "Logged out successfully. Remove the token from client storage."}
