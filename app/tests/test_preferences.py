# app/tests/test_preferences.py
"""Tests for user preferences endpoints."""

import pytest


@pytest.mark.asyncio
async def test_get_preferences_default(auth_client):
    """New user should have default preferences."""
    resp = await auth_client.get("/preferences")
    assert resp.status_code == 200
    prefs = resp.json()
    assert prefs["preferred_model"] == "default"
    assert prefs["theme"] == "light"
    assert prefs["top_k"] == 4


@pytest.mark.asyncio
async def test_update_preferences(auth_client):
    resp = await auth_client.put("/preferences", json={
        "theme": "dark",
        "top_k": 8,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["theme"] == "dark"
    assert data["top_k"] == 8

    # Verify persistence
    resp = await auth_client.get("/preferences")
    assert resp.json()["theme"] == "dark"
    assert resp.json()["top_k"] == 8


@pytest.mark.asyncio
async def test_update_preferred_model(auth_client):
    resp = await auth_client.put("/preferences", json={
        "preferred_model": "gpt-4",
    })
    assert resp.status_code == 200
    assert resp.json()["preferred_model"] == "gpt-4"


@pytest.mark.asyncio
async def test_preferences_requires_auth(client):
    resp = await client.get("/preferences")
    assert resp.status_code == 401

    resp = await client.put("/preferences", json={"theme": "dark"})
    assert resp.status_code == 401
