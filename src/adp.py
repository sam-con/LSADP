"""ADP data loading and player matching."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models import ConfigError
from src.utils import ensure_columns, normalize_player_name, rank_players_within_position


class ADPDataProvider:
    """CSV-backed ADP provider."""

    required_columns = ("player_name", "position", "team", "adp")

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise ConfigError(f"ADP file not found at {self.path}")
        frame = pd.read_csv(self.path)
        ensure_columns(frame, self.required_columns)
        if frame.empty:
            raise ConfigError(f"ADP file at {self.path} is empty")
        frame = frame.copy()
        if "player_id" not in frame.columns:
            frame["player_id"] = pd.NA
        frame["player_id"] = frame["player_id"].astype("string")
        frame["player_name"] = frame["player_name"].astype(str)
        frame["position"] = frame["position"].astype(str).str.upper()
        frame["team"] = frame["team"].astype(str)
        frame["adp"] = frame["adp"].astype(float)
        frame["normalized_name"] = frame["player_name"].map(normalize_player_name)
        return rank_players_within_position(frame)


def match_players_by_identity(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Match player tables using player_id first and normalized name second."""

    right_columns = [column for column in right.columns if column not in {"player_name", "team", "position"}]
    if left["player_id"].notna().any() and right["player_id"].notna().any():
        merged = left.merge(right[["player_id", *right_columns]].drop_duplicates("player_id"), on="player_id", how="left")
    else:
        merged = left.merge(
            right[["normalized_name", "position", *right_columns]].drop_duplicates(["normalized_name", "position"]),
            on=["normalized_name", "position"],
            how="left",
        )
    return merged

