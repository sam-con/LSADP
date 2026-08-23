"""Cached Sleeper API access with response validation."""

from __future__ import annotations

import re
from typing import Any

import requests
import streamlit as st

API_BASE = "https://api.sleeper.app"
PROJECTIONS_URL = f"{API_BASE}/projections/nfl"
_VALID_ID = re.compile(r"^[A-Za-z0-9_-]{6,80}$")


class SleeperAPIError(RuntimeError):
    pass


def validate_draft_id(draft_id: str) -> str:
    value = (draft_id or "").strip()
    if not _VALID_ID.fullmatch(value):
        raise ValueError("Enter a valid Sleeper draft ID (letters, numbers, underscores, or hyphens).")
    return value


def _get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    try:
        response = requests.get(url, params=params, timeout=25)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise SleeperAPIError(f"Sleeper API request failed: {exc}") from exc
    except ValueError as exc:
        raise SleeperAPIError("Sleeper returned an invalid JSON response.") from exc


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_draft(draft_id: str) -> dict:
    data = _get_json(f"{API_BASE}/v1/draft/{validate_draft_id(draft_id)}")
    if not isinstance(data, dict) or not data.get("league_id"):
        raise SleeperAPIError("Draft was not found or is not associated with a league.")
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_league(league_id: str) -> dict:
    data = _get_json(f"{API_BASE}/v1/league/{league_id}")
    if not isinstance(data, dict) or not data.get("league_id"):
        raise SleeperAPIError("League settings were not found.")
    return data


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_projections(season: int) -> list[dict]:
    data = _get_json(f"{PROJECTIONS_URL}/{season}", {"season_type": "regular", "order_by": "pts_ppr"})
    if not isinstance(data, list):
        raise SleeperAPIError("Sleeper returned projections in an unexpected format.")
    return data


def fetch_draft_league(draft_id: str) -> tuple[dict, dict]:
    draft = fetch_draft(draft_id)
    return draft, fetch_league(str(draft["league_id"]))
