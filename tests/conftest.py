from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest


def make_league_payload(
    league_id: str,
    season: int,
    previous_league_id: str | None,
    scoring_settings: dict[str, float] | None = None,
    roster_positions: list[str] | None = None,
    total_rosters: int = 12,
    playoff_week_start: int = 15,
) -> dict[str, Any]:
    return {
        "league_id": league_id,
        "name": f"League {season}",
        "season": str(season),
        "total_rosters": total_rosters,
        "scoring_settings": scoring_settings or {"rec": 1.0, "pass_td": 4.0, "rec_te": 1.0},
        "roster_positions": roster_positions
        or ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN", "BN", "BN", "BN", "BN"],
        "previous_league_id": previous_league_id,
        "settings": {"playoff_week_start": playoff_week_start},
    }


def generate_players_payload() -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for prefix, position, total in [("Q", "QB", 8), ("R", "RB", 12), ("W", "WR", 12), ("T", "TE", 8)]:
        for index in range(1, total + 1):
            player_id = f"{prefix}{index}"
            payload[player_id] = {
                "player_id": player_id,
                "full_name": f"{position} Player {index}",
                "position": position,
                "team": f"T{index}",
            }
    return payload


def _position_base_points(position: str, rank_index: int) -> float:
    if position == "QB":
        return 26.0 - 1.25 * rank_index
    if position == "RB":
        return 22.5 - 0.95 * rank_index
    if position == "WR":
        return 21.0 - 0.8 * rank_index
    return 16.5 - 0.7 * rank_index


def generate_week_matchups(players_payload: dict[str, dict[str, Any]], season: int, week: int) -> list[dict[str, Any]]:
    return generate_week_matchups_for_profile(players_payload, season, week, profile="ppr")


def generate_week_matchups_for_profile(
    players_payload: dict[str, dict[str, Any]],
    season: int,
    week: int,
    profile: str,
) -> list[dict[str, Any]]:
    ordered_ids = sorted(players_payload)
    first_half = ordered_ids[: len(ordered_ids) // 2]
    second_half = ordered_ids[len(ordered_ids) // 2 :]
    matchups = []
    profile_bonus = {
        "standard": {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0},
        "half_ppr": {"QB": 0.0, "RB": 0.6, "WR": 1.1, "TE": 1.0},
        "ppr": {"QB": 0.0, "RB": 1.2, "WR": 2.2, "TE": 1.9},
        "sf_standard": {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0},
        "sf_half_ppr": {"QB": 0.0, "RB": 0.6, "WR": 1.1, "TE": 1.0},
        "sf_ppr": {"QB": 0.0, "RB": 1.2, "WR": 2.2, "TE": 1.9},
    }
    for roster_id, ids in enumerate([first_half, second_half], start=1):
        players_points: dict[str, float] = {}
        starters: list[str] = []
        position_counter: dict[str, int] = {}
        for player_id in ids:
            position = players_payload[player_id]["position"]
            rank_index = int(player_id[1:]) - 1
            season_bonus = 0.25 * (season - 2022)
            week_bonus = 0.1 * (week % 3)
            scoring_bonus = profile_bonus[profile][position]
            players_points[player_id] = round(_position_base_points(position, rank_index) + scoring_bonus + season_bonus + week_bonus, 2)
            position_counter.setdefault(position, 0)
            position_counter[position] += 1
            if (
                (position == "QB" and position_counter[position] <= 1)
                or (position == "RB" and position_counter[position] <= 2)
                or (position == "WR" and position_counter[position] <= 2)
                or (position == "TE" and position_counter[position] <= 1)
            ):
                starters.append(player_id)
        matchups.append(
            {
                "roster_id": roster_id,
                "players": ids,
                "starters": starters,
                "players_points": players_points,
                "starters_points": [players_points[player_id] for player_id in starters],
            }
        )
    return matchups


def build_player_weeks_dataframe(players_payload: dict[str, dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for season in range(2022, 2026):
        for week in range(1, 11):
            for matchup in generate_week_matchups(players_payload, season=season, week=week):
                for player_id, points in matchup["players_points"].items():
                    records.append(
                        {
                            "season": season,
                            "week": week,
                            "league_id": str(season),
                            "player_id": player_id,
                            "player_name": players_payload[player_id]["full_name"],
                            "position": players_payload[player_id]["position"],
                            "team": players_payload[player_id]["team"],
                            "fantasy_points": points,
                            "starter_flag": player_id in matchup["starters"],
                            "roster_id": matchup["roster_id"],
                        }
                    )
    return pd.DataFrame(records)


@dataclass
class MockSleeperClient:
    leagues: dict[str, dict[str, Any]]
    matchups: dict[tuple[str, int], list[dict[str, Any]]]
    players_payload: dict[str, dict[str, Any]]
    users: dict[str, dict[str, Any]] | None = None
    user_leagues: dict[tuple[str, int], list[str]] | None = None
    league_users: dict[str, list[dict[str, Any]]] | None = None

    def get_league(self, league_id: str):
        from src.models import SleeperAPIError
        from src.sleeper import parse_league_settings

        if league_id not in self.leagues:
            raise SleeperAPIError(f"Unknown league {league_id}")
        return parse_league_settings(self.leagues[league_id])

    def get_matchups(self, league_id: str, week: int):
        return self.matchups[(league_id, week)]

    def get_players(self, sport: str = "nfl"):
        return self.players_payload

    def get_user(self, username_or_user_id: str):
        from src.models import SleeperAPIError

        users = self.users or {}
        if username_or_user_id not in users:
            raise SleeperAPIError(f"Unknown user {username_or_user_id}")
        return users[username_or_user_id]

    def get_user_leagues(self, user_id: str, sport: str = "nfl", season: int | str = 2026):
        from src.models import SleeperAPIError

        league_ids = (self.user_leagues or {}).get((str(user_id), int(season)))
        if league_ids is None:
            raise SleeperAPIError(f"Unknown leagues for user {user_id} season {season}")
        return [self.get_league(league_id) for league_id in league_ids]

    def get_league_users(self, league_id: str):
        return (self.league_users or {}).get(str(league_id), [])


@pytest.fixture
def players_payload() -> dict[str, dict[str, Any]]:
    return generate_players_payload()


@pytest.fixture
def mock_client(players_payload: dict[str, dict[str, Any]]) -> MockSleeperClient:
    base_scoring = {"rec": 1.0, "pass_td": 4.0, "rec_te": 1.0}
    half_scoring = {"rec": 0.5, "pass_td": 4.0, "rec_te": 0.5}
    standard_scoring = {"rec": 0.0, "pass_td": 4.0, "rec_te": 0.0}
    baseline_positions = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN", "BN", "BN", "BN", "BN"]
    leagues = {
        "2026": make_league_payload("2026", 2026, "2025", base_scoring, baseline_positions),
        "2025": make_league_payload("2025", 2025, "2024", base_scoring, baseline_positions),
        "2024": make_league_payload("2024", 2024, "2023", base_scoring, baseline_positions),
        "2023": make_league_payload("2023", 2023, "2022", base_scoring, baseline_positions),
        "2022": make_league_payload("2022", 2022, None, base_scoring, baseline_positions),
        "sf2026": make_league_payload(
            "sf2026",
            2026,
            "sf2025",
            base_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sf2025": make_league_payload(
            "sf2025",
            2025,
            "sf2024",
            base_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sf2024": make_league_payload(
            "sf2024",
            2024,
            "sf2023",
            base_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sf2023": make_league_payload(
            "sf2023",
            2023,
            "sf2022",
            base_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sf2022": make_league_payload(
            "sf2022",
            2022,
            None,
            base_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "half2026": make_league_payload("half2026", 2026, "half2025", half_scoring, baseline_positions),
        "half2025": make_league_payload("half2025", 2025, "half2024", half_scoring, baseline_positions),
        "half2024": make_league_payload("half2024", 2024, "half2023", half_scoring, baseline_positions),
        "half2023": make_league_payload("half2023", 2023, "half2022", half_scoring, baseline_positions),
        "half2022": make_league_payload("half2022", 2022, None, half_scoring, baseline_positions),
        "std2026": make_league_payload("std2026", 2026, "std2025", standard_scoring, baseline_positions),
        "std2025": make_league_payload("std2025", 2025, "std2024", standard_scoring, baseline_positions),
        "std2024": make_league_payload("std2024", 2024, "std2023", standard_scoring, baseline_positions),
        "std2023": make_league_payload("std2023", 2023, "std2022", standard_scoring, baseline_positions),
        "std2022": make_league_payload("std2022", 2022, None, standard_scoring, baseline_positions),
        "sfhalf2026": make_league_payload(
            "sfhalf2026",
            2026,
            "sfhalf2025",
            half_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sfhalf2025": make_league_payload(
            "sfhalf2025",
            2025,
            "sfhalf2024",
            half_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sfhalf2024": make_league_payload(
            "sfhalf2024",
            2024,
            "sfhalf2023",
            half_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sfhalf2023": make_league_payload(
            "sfhalf2023",
            2023,
            "sfhalf2022",
            half_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sfhalf2022": make_league_payload(
            "sfhalf2022",
            2022,
            None,
            half_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sfstd2026": make_league_payload(
            "sfstd2026",
            2026,
            "sfstd2025",
            standard_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sfstd2025": make_league_payload(
            "sfstd2025",
            2025,
            "sfstd2024",
            standard_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sfstd2024": make_league_payload(
            "sfstd2024",
            2024,
            "sfstd2023",
            standard_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sfstd2023": make_league_payload(
            "sfstd2023",
            2023,
            "sfstd2022",
            standard_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
        "sfstd2022": make_league_payload(
            "sfstd2022",
            2022,
            None,
            standard_scoring,
            ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX", "BN", "BN", "BN", "BN", "BN"],
        ),
    }
    matchups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    profiles = {
        "2022": "ppr",
        "2023": "ppr",
        "2024": "ppr",
        "2025": "ppr",
        "sf2022": "sf_ppr",
        "sf2023": "sf_ppr",
        "sf2024": "sf_ppr",
        "sf2025": "sf_ppr",
        "half2022": "half_ppr",
        "half2023": "half_ppr",
        "half2024": "half_ppr",
        "half2025": "half_ppr",
        "std2022": "standard",
        "std2023": "standard",
        "std2024": "standard",
        "std2025": "standard",
        "sfhalf2022": "sf_half_ppr",
        "sfhalf2023": "sf_half_ppr",
        "sfhalf2024": "sf_half_ppr",
        "sfhalf2025": "sf_half_ppr",
        "sfstd2022": "sf_standard",
        "sfstd2023": "sf_standard",
        "sfstd2024": "sf_standard",
        "sfstd2025": "sf_standard",
    }
    for season_key, profile in profiles.items():
        season = int(season_key[-4:])
        for week in range(1, 15):
            matchups[(season_key, week)] = generate_week_matchups_for_profile(players_payload, season=season, week=week, profile=profile)
    return MockSleeperClient(leagues=leagues, matchups=matchups, players_payload=players_payload)


@pytest.fixture
def player_weeks_df(players_payload: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return build_player_weeks_dataframe(players_payload)


@pytest.fixture
def adp_frame(players_payload: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    current_adp = 1.0
    for position in ["RB", "WR", "QB", "TE"]:
        ids = [player_id for player_id, payload in players_payload.items() if payload["position"] == position][:6]
        for player_id in ids:
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": players_payload[player_id]["full_name"],
                    "position": position,
                    "team": players_payload[player_id]["team"],
                    "adp": current_adp,
                }
            )
            current_adp += 1.5
    return pd.DataFrame(rows)


def generate_canonical_adp_frame(players_payload: dict[str, dict[str, Any]], environment_key: str) -> pd.DataFrame:
    reception_bonus = {
        "1qb_standard": {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0},
        "1qb_half_ppr": {"QB": 0.0, "RB": 2.0, "WR": 4.0, "TE": 3.0},
        "1qb_ppr": {"QB": 0.0, "RB": 4.0, "WR": 8.0, "TE": 6.0},
        "sf_standard": {"QB": 0.0, "RB": 0.0, "WR": 0.0, "TE": 0.0},
        "sf_half_ppr": {"QB": 0.0, "RB": 2.0, "WR": 4.0, "TE": 3.0},
        "sf_ppr": {"QB": 0.0, "RB": 4.0, "WR": 8.0, "TE": 6.0},
    }
    qb_bonus = 28.0 if environment_key.startswith("sf_") else 0.0
    rows = []
    for position in ["QB", "RB", "WR", "TE"]:
        ids = [player_id for player_id, payload in players_payload.items() if payload["position"] == position][:6]
        for index, player_id in enumerate(ids, start=1):
            base_value = {
                "QB": 70.0 - 4.5 * index,
                "RB": 108.0 - 5.0 * index,
                "WR": 102.0 - 4.4 * index,
                "TE": 78.0 - 4.2 * index,
            }[position]
            score = base_value + reception_bonus[environment_key][position] + (qb_bonus if position == "QB" else 0.0)
            rows.append(
                {
                    "player_id": player_id,
                    "player_name": players_payload[player_id]["full_name"],
                    "position": position,
                    "team": players_payload[player_id]["team"],
                    "score": score,
                }
            )
    ordered = sorted(rows, key=lambda row: (-row["score"], row["player_name"]))
    adp_rows = []
    current_adp = 1.0
    for row in ordered:
        adp_rows.append(
            {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "position": row["position"],
                "team": row["team"],
                "adp": round(current_adp, 2),
            }
        )
        current_adp += 1.35
    return pd.DataFrame(adp_rows)


@pytest.fixture
def canonical_league_ids() -> dict[str, str]:
    return {
        "1qb_standard": "std2026",
        "1qb_half_ppr": "half2026",
        "1qb_ppr": "2026",
        "sf_standard": "sfstd2026",
        "sf_half_ppr": "sfhalf2026",
        "sf_ppr": "sf2026",
    }


@pytest.fixture
def canonical_adp_frames(players_payload: dict[str, dict[str, Any]]) -> dict[str, pd.DataFrame]:
    return {
        environment_key: generate_canonical_adp_frame(players_payload, environment_key)
        for environment_key in [
            "1qb_standard",
            "1qb_half_ppr",
            "1qb_ppr",
            "sf_standard",
            "sf_half_ppr",
            "sf_ppr",
        ]
    }


@pytest.fixture
def canonical_adp_paths(tmp_path, canonical_adp_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    paths = {}
    for environment_key, frame in canonical_adp_frames.items():
        path = tmp_path / f"{environment_key}.csv"
        frame.to_csv(path, index=False)
        paths[environment_key] = path
    return paths
