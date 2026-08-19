"""Find donor Sleeper leagues with valid historical chains and similar settings."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.canonical import canonical_environment_key_for_league
from src.sleeper import SleeperClient
from src.sleeper_history import load_league_chain, select_required_history
from src.utils import material_scoring_differences


TODAY = date(2026, 8, 19)


def roster_slot_counts(roster_positions: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for slot in roster_positions:
        counts[slot] = counts.get(slot, 0) + 1
    return counts


def score_candidate(target, candidate) -> tuple[int, int, int]:
    target_scoring = target.scoring_settings
    candidate_scoring = candidate.scoring_settings
    scoring_diffs = material_scoring_differences(target_scoring, candidate_scoring)
    roster_target = roster_slot_counts(target.roster_positions)
    roster_candidate = roster_slot_counts(candidate.roster_positions)
    roster_penalty = sum(
        abs(roster_target.get(slot, 0) - roster_candidate.get(slot, 0))
        for slot in set(roster_target) | set(roster_candidate)
    )
    team_penalty = abs(int(target.total_rosters) - int(candidate.total_rosters))
    return (len(scoring_diffs), roster_penalty, team_penalty)


def is_exact_match(target, candidate) -> bool:
    return score_candidate(target, candidate) == (0, 0, 0)


def describe_candidate(target, candidate, chain: list[Any], username: str) -> dict[str, Any]:
    scoring_diffs = material_scoring_differences(target.scoring_settings, candidate.scoring_settings)
    return {
        "username": username,
        "league_id": candidate.league_id,
        "name": candidate.name,
        "season": candidate.season,
        "team_count": candidate.total_rosters,
        "target_environment": canonical_environment_key_for_league(target),
        "candidate_environment": canonical_environment_key_for_league(candidate),
        "same_environment": canonical_environment_key_for_league(target) == canonical_environment_key_for_league(candidate),
        "same_team_count": candidate.total_rosters == target.total_rosters,
        "scoring_difference_count": len(scoring_diffs),
        "scoring_differences": scoring_diffs,
        "target_roster_positions": target.roster_positions,
        "candidate_roster_positions": candidate.roster_positions,
        "historical_chain_seasons": [league.season for league in chain],
        "score": score_candidate(target, candidate),
    }


def find_candidates(usernames: list[str], target_league_id: str, season: int, limit: int) -> list[dict[str, Any]]:
    client = SleeperClient()
    target = client.get_league(target_league_id)
    matches: list[dict[str, Any]] = []

    for username in usernames:
        user = client.get_user(username)
        user_id = str(user["user_id"])
        leagues = client.get_user_leagues(user_id=user_id, season=season)
        for league in leagues:
            try:
                chain = load_league_chain(client, league.league_id)
                select_required_history(chain, today=TODAY)
            except Exception:
                continue
            matches.append(describe_candidate(target, league, chain, username))

    matches.sort(key=lambda item: (item["score"], item["same_environment"] is False, item["league_id"]))
    return matches[:limit]


def print_summary(payload: dict[str, Any], exact_only: bool) -> None:
    print(
        f"Target league: {payload['target_league_id']} | season scanned: {payload['season']} | "
        f"candidates found: {payload['candidate_count']}"
    )
    print("")
    if not payload["matches"]:
        print("No donor candidates found.")
        return

    if exact_only:
        print("Exact matches only: scoring, roster slots, and team count must all match.")
        print("")
    for index, match in enumerate(payload["matches"], start=1):
        scoring_penalty, roster_penalty, team_penalty = match["score"]
        print(
            f"{index:>2}. {match['username']} | {match['league_id']} | {match['name']} | "
            f"env={match['candidate_environment']} | teams={match['team_count']} | "
            f"score=({scoring_penalty}, {roster_penalty}, {team_penalty})"
        )
        print(
            f"    same_env={match['same_environment']} | same_team_count={match['same_team_count']} | "
            f"history={match['historical_chain_seasons']}"
        )
        if match["scoring_differences"]:
            print(f"    scoring_differences={match['scoring_differences']}")
        else:
            print("    scoring_differences=[]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find donor Sleeper leagues with a valid four-season chain.")
    parser.add_argument("--target-league-id", required=True, help="League ID whose settings should be matched.")
    parser.add_argument(
        "--usernames",
        required=True,
        nargs="+",
        help="One or more Sleeper usernames to scan for candidate donor leagues.",
    )
    parser.add_argument("--season", type=int, default=2026, help="Sleeper league season to inspect. Default: 2026.")
    parser.add_argument("--limit", type=int, default=25, help="Maximum number of candidates to print.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--json", action="store_true", help="Print full JSON to stdout instead of a compact summary.")
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Only return leagues whose scoring, roster slots, and team count exactly match the target league.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matches = find_candidates(
        usernames=args.usernames,
        target_league_id=args.target_league_id,
        season=args.season,
        limit=args.limit,
    )
    if args.exact_only:
        matches = [match for match in matches if tuple(match["score"]) == (0, 0, 0)]
    payload = {
        "target_league_id": args.target_league_id,
        "season": args.season,
        "candidate_count": len(matches),
        "matches": matches,
    }
    text = json.dumps(payload, indent=2)
    if args.json:
        print(text)
    else:
        print_summary(payload, exact_only=args.exact_only)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
