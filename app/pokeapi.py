# app/pokeapi.py
"""Async client for the PokeAPI (https://pokeapi.co) -- free, no auth required."""

from typing import Any

import httpx

from .http_client import get_http_client

POKEAPI_URL = "https://pokeapi.co/api/v2/pokemon"


async def get_pokemon(name: str) -> dict[str, Any]:
    """
    Fetch data for a Pokemon by *name* or Pokedex ID.

    Returns a dict with structured Pokemon data on success,
    or a dict with an ``error`` key on failure.
    """
    slug = name.lower().strip()
    if not slug:
        return {"error": "Please enter a Pokemon name or Pokedex number."}

    try:
        async with get_http_client(timeout=10) as client:
            resp = await client.get(f"{POKEAPI_URL}/{slug}")

        if resp.status_code == 404:
            return {"error": f"Pokemon '{name}' not found. Check the spelling or try a Pokedex number."}

        resp.raise_for_status()
        data = resp.json()

        stats = {}
        for s in data.get("stats", []):
            stat_name = s["stat"]["name"]
            stats[stat_name] = s["base_stat"]

        return {
            "name": data["name"].capitalize(),
            "id": data["id"],
            "types": [t["type"]["name"].capitalize() for t in data.get("types", [])],
            "height_m": data["height"] / 10,
            "weight_kg": data["weight"] / 10,
            "stats": stats,
            "abilities": [a["ability"]["name"].replace("-", " ").capitalize() for a in data.get("abilities", [])],
            "sprite": data["sprites"].get("front_default", ""),
        }

    except httpx.HTTPStatusError as exc:
        return {"error": f"PokeAPI returned HTTP {exc.response.status_code}."}
    except httpx.RequestError as exc:
        return {"error": f"Network error contacting PokeAPI: {exc}"}
