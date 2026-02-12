# app/tests/test_auth.py
"""Tests for authentication: register, login, logout, and token validation."""

import pytest
from app.auth import hash_password, verify_password


# ── Unit tests: password hashing ──────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_and_verify(self):
        pw = "my_secure_password"
        hashed = hash_password(pw)
        assert hashed != pw
        assert verify_password(pw, hashed)

    def test_wrong_password_fails(self):
        hashed = hash_password("correct_password")
        assert not verify_password("wrong_password", hashed)

    def test_different_hashes_for_same_password(self):
        """bcrypt salts should produce different hashes each time."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2
        assert verify_password("same", h1)
        assert verify_password("same", h2)


# ── Integration tests: register endpoint ──────────────────────────────────

@pytest.mark.asyncio
async def test_register_success(client):
    resp = await client.post("/auth/register", json={
        "username": "newuser",
        "password": "password123",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert "token" in data
    assert data["username"] == "newuser"
    assert len(data["token"]) == 32  # uuid4 hex


@pytest.mark.asyncio
async def test_register_duplicate_username(client):
    await client.post("/auth/register", json={
        "username": "dupeuser",
        "password": "password123",
    })
    resp = await client.post("/auth/register", json={
        "username": "dupeuser",
        "password": "anotherpass",
    })
    assert resp.status_code == 409
    assert "already taken" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_short_password(client):
    resp = await client.post("/auth/register", json={
        "username": "shortpw",
        "password": "12345",
    })
    assert resp.status_code == 422  # validation error


@pytest.mark.asyncio
async def test_register_empty_username(client):
    resp = await client.post("/auth/register", json={
        "username": "   ",
        "password": "password123",
    })
    assert resp.status_code == 422


# ── Integration tests: login endpoint ─────────────────────────────────────

@pytest.mark.asyncio
async def test_login_success(client):
    # Register first
    await client.post("/auth/register", json={
        "username": "loginuser",
        "password": "password123",
    })
    # Login
    resp = await client.post("/auth/login", json={
        "username": "loginuser",
        "password": "password123",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "token" in data
    assert data["username"] == "loginuser"


@pytest.mark.asyncio
async def test_login_wrong_password(client):
    await client.post("/auth/register", json={
        "username": "wrongpw_user",
        "password": "password123",
    })
    resp = await client.post("/auth/login", json={
        "username": "wrongpw_user",
        "password": "wrongpassword",
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_user(client):
    resp = await client.post("/auth/login", json={
        "username": "ghost",
        "password": "password123",
    })
    assert resp.status_code == 401


# ── Integration tests: auth-protected routes ──────────────────────────────

@pytest.mark.asyncio
async def test_protected_route_no_token(client):
    resp = await client.get("/favorites")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_invalid_token(client):
    resp = await client.get("/favorites", headers={
        "Authorization": "Bearer invalid_token_123",
    })
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_protected_route_valid_token(auth_client):
    resp = await auth_client.get("/favorites")
    assert resp.status_code == 200
