"""Turn raw Sleeper season projection records into an offensive player table."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from .models import CORE_POSITIONS


def projection_records_to_frame(records: Iterable[dict], adp_field: str) -> pd.DataFrame:
    rows: list[dict] = []
    for record in records:
        player = record.get("player") or {}
        stats = record.get("stats") or {}
        position = player.get("position")
        if position not in CORE_POSITIONS:
            continue
        # Season records can include deep players; retain anyone with a projected stat or market ADP.
        has_projection = any(key in stats for key in ("pass_yd", "rush_yd", "rec", "rec_yd", "rec_td"))
        raw_adp = stats.get(adp_field)
        try:
            adp = float(raw_adp)
        except (TypeError, ValueError):
            adp = math.nan
        if not has_projection and (math.isnan(adp) or adp >= 999):
            continue
        first = player.get("first_name") or ""
        last = player.get("last_name") or ""
        name = f"{first} {last}".strip() or str(record.get("player_id", "Unknown"))
        rows.append(
            {
                "player_id": str(record.get("player_id", "")),
                "player": name,
                "team": player.get("team") or record.get("team") or "FA",
                "position": position,
                "current_adp": adp,
                "stats": stats,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["player_id", "player", "team", "position", "current_adp", "stats"])
    frame = pd.DataFrame(rows).drop_duplicates("player_id", keep="first")
    # Sleeper uses 999/null for unavailable ADP. Give unmarketed projected players a deterministic
    # tail value so they remain visible without claiming a real market ADP.
    known = frame["current_adp"].where(frame["current_adp"].between(0.01, 998))
    tail_start = max(float(known.max()) if known.notna().any() else 0, 300.0)
    missing = ~frame["current_adp"].between(0.01, 998)
    fallback_order = frame.loc[missing, "stats"].map(lambda s: float(s.get("pts_ppr", 0) or 0)).rank(method="first", ascending=False)
    frame.loc[missing, "current_adp"] = tail_start + fallback_order
    frame["market_adp_available"] = ~missing
    return frame
