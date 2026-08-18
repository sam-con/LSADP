"""Historical league-chain traversal and scoring-consistency validation."""

from __future__ import annotations

from datetime import date

from src.models import HistoricalLeagueSummary, LeagueSettings, ScoringConsistencyError, ScoringDifference, SleeperAPIError
from src.sleeper import SleeperClient, as_historical_summary
from src.utils import material_scoring_differences, normalize_scoring_settings, required_completed_seasons


def load_league_chain(client: SleeperClient, league_id: str, max_depth: int = 8) -> list[HistoricalLeagueSummary]:
    """Follow `previous_league_id` backward starting from the supplied league."""

    seen: set[str] = set()
    chain: list[HistoricalLeagueSummary] = []
    current_id: str | None = league_id
    depth = 0

    while current_id and depth < max_depth:
        if current_id in seen:
            break
        seen.add(current_id)
        try:
            league = client.get_league(current_id)
        except SleeperAPIError:
            break
        chain.append(as_historical_summary(league))
        current_id = league.previous_league_id
        depth += 1

    return chain


def select_required_history(
    chain: list[HistoricalLeagueSummary],
    today: date | None = None,
    window: int = 4,
) -> list[HistoricalLeagueSummary]:
    """Select exactly the previous completed NFL seasons from the league chain."""

    required_seasons = required_completed_seasons(today=today, window=window)
    by_season = {league.season: league for league in chain}
    missing = [season for season in required_seasons if season not in by_season]
    if missing:
        missing_text = ", ".join(str(season) for season in missing)
        raise ScoringConsistencyError(f"This league does not have four completed Sleeper seasons available. Missing: {missing_text}")
    return [by_season[season] for season in required_seasons]


def compare_scoring_across_history(
    current_league: LeagueSettings,
    historical_leagues: list[HistoricalLeagueSummary],
) -> list[ScoringDifference]:
    """Return material scoring differences between current and historical seasons."""

    current_settings = normalize_scoring_settings(current_league.scoring_settings)
    differences: list[ScoringDifference] = []
    for league in historical_leagues:
        for key, current_value, historical_value in material_scoring_differences(current_settings, league.scoring_settings):
            differences.append(
                ScoringDifference(
                    season=league.season,
                    key=key,
                    current_value=current_value,
                    historical_value=historical_value,
                )
            )
    return differences


def validate_scoring_consistency(
    current_league: LeagueSettings,
    historical_leagues: list[HistoricalLeagueSummary],
) -> None:
    """Raise when scoring settings changed across the required historical window."""

    differences = compare_scoring_across_history(current_league, historical_leagues)
    if not differences:
        return
    lines = [
        f"{difference.season}: {difference.key} {difference.historical_value} -> {difference.current_value}"
        for difference in differences
    ]
    joined = "; ".join(lines)
    raise ScoringConsistencyError(
        "Historical scoring settings changed across this league's previous four seasons, "
        f"so a reliable league-specific ADP cannot be generated with the current model. Differences: {joined}"
    )
