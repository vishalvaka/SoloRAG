# app/tests/test_favorites.py
"""Tests for favorites CRUD endpoints."""

import pytest


@pytest.mark.asyncio
async def test_favorites_empty(auth_client):
    resp = await auth_client.get("/favorites")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_create_favorite(auth_client):
    resp = await auth_client.post("/favorites", json={
        "question": "What is a payout?",
        "answer": "A transfer of funds to your bank account.",
        "sources": [{"text": "some source", "score": 0.9}],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["question"] == "What is a payout?"
    assert data["answer"] == "A transfer of funds to your bank account."
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_list_favorites(auth_client):
    # Create two favorites
    await auth_client.post("/favorites", json={
        "question": "Q1", "answer": "A1",
    })
    await auth_client.post("/favorites", json={
        "question": "Q2", "answer": "A2",
    })

    resp = await auth_client.get("/favorites")
    assert resp.status_code == 200
    favs = resp.json()
    assert len(favs) == 2


@pytest.mark.asyncio
async def test_delete_favorite(auth_client):
    # Create a favorite
    resp = await auth_client.post("/favorites", json={
        "question": "Delete me", "answer": "Bye",
    })
    fav_id = resp.json()["id"]

    # Delete it
    resp = await auth_client.delete(f"/favorites/{fav_id}")
    assert resp.status_code == 204

    # Verify it's gone
    resp = await auth_client.get("/favorites")
    assert resp.json() == []


@pytest.mark.asyncio
async def test_delete_nonexistent_favorite(auth_client):
    resp = await auth_client.delete("/favorites/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_favorites_requires_auth(client):
    resp = await client.get("/favorites")
    assert resp.status_code == 401

    resp = await client.post("/favorites", json={
        "question": "Q", "answer": "A",
    })
    assert resp.status_code == 401
