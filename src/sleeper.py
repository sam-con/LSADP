"""Sleeper API client and response parsing."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import requests

from src.config import REQUEST_TIMEOUT_SECONDS
from src.models import HistoricalLeagueSummary, LeagueSettings, SleeperAPIError
from src.utils import normalize_scoring_settings


class SleeperClient:
    """Thin official Sleeper API client."""

    base_url = "https://api.sleeper.app/v1"

    def __init__(self, timeout_seconds: int = REQUEST_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def _get_json(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.get(url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SleeperAPIError(f"Unable to load Sleeper data from {url}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise SleeperAPIError(f"Sleeper returned malformed JSON for {url}") from exc

    def get_league(self, league_id: str) -> LeagueSettings:
        payload = self._get_json(f"/league/{league_id}")
        return parse_league_settings(payload)

    def get_user(self, username_or_user_id: str) -> dict[str, Any]:
        payload = self._get_json(f"/user/{username_or_user_id}")
        if not isinstance(payload, dict):
            raise SleeperAPIError(f"Sleeper returned malformed user data for {username_or_user_id}")
        return payload

    def get_user_leagues(self, user_id: str, sport: str = "nfl", season: int | str = 2026) -> list[LeagueSettings]:
        payload = self._get_json(f"/user/{user_id}/leagues/{sport}/{season}")
        if not isinstance(payload, list):
            raise SleeperAPIError(f"Sleeper returned malformed league list for user {user_id}, season {season}")
        return [parse_league_settings(item) for item in payload]

    def get_league_users(self, league_id: str) -> list[dict[str, Any]]:
        payload = self._get_json(f"/league/{league_id}/users")
        if not isinstance(payload, list):
            raise SleeperAPIError(f"Sleeper returned malformed user list for league {league_id}")
        return payload

    def get_matchups(self, league_id: str, week: int) -> list[dict[str, Any]]:
        payload = self._get_json(f"/league/{league_id}/matchups/{week}")
        if not isinstance(payload, list):
            raise SleeperAPIError(f"Sleeper returned malformed matchup data for league {league_id}, week {week}")
        return payload

    def get_players(self, sport: str = "nfl") -> dict[str, Any]:
        payload = self._get_json(f"/players/{sport}")
        if not isinstance(payload, dict):
            raise SleeperAPIError("Sleeper returned malformed player metadata")
        return payload


def parse_league_settings(payload: dict[str, Any]) -> LeagueSettings:
    """Parse a league payload into the internal settings model."""

    if not isinstance(payload, dict):
        raise SleeperAPIError("Sleeper league response was not a JSON object")

    required = ("league_id", "name", "season", "total_rosters", "scoring_settings", "roster_positions")
    missing = [field for field in required if field not in payload]
    if missing:
        raise SleeperAPIError(f"League response missing required fields: {', '.join(missing)}")

    scoring_settings = payload.get("scoring_settings")
    roster_positions = payload.get("roster_positions")
    if not isinstance(scoring_settings, dict) or not isinstance(roster_positions, list):
        raise SleeperAPIError("League response contains malformed scoring settings or roster positions")

    settings_payload = payload.get("settings", {}) if isinstance(payload.get("settings"), dict) else {}
    playoff_week_start = settings_payload.get("playoff_week_start")

    return LeagueSettings(
        league_id=str(payload["league_id"]),
        name=str(payload["name"]),
        season=int(payload["season"]),
        total_rosters=int(payload["total_rosters"]),
        scoring_settings=normalize_scoring_settings(scoring_settings),
        roster_positions=[str(slot) for slot in roster_positions],
        previous_league_id=str(payload["previous_league_id"]) if payload.get("previous_league_id") else None,
        playoff_week_start=int(playoff_week_start) if playoff_week_start else None,
    )


def as_historical_summary(league: LeagueSettings) -> HistoricalLeagueSummary:
    """Convert current settings into a slimmer historical summary."""

    payload = asdict(league)
    return HistoricalLeagueSummary(
        league_id=payload["league_id"],
        season=payload["season"],
        scoring_settings=payload["scoring_settings"],
        roster_positions=payload["roster_positions"],
        total_rosters=payload["total_rosters"],
        previous_league_id=payload["previous_league_id"],
        playoff_week_start=payload["playoff_week_start"],
    )
