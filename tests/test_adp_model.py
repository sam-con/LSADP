import pandas as pd

from src.adp_model import estimate_adjusted_adp


REFERENCE_ROSTER = ("QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX")


def _players():
    rows = []
    for position, scores in {"QB": [300, 280, 260, 240, 220, 200], "RB": [250, 220, 195, 170, 145, 120], "WR": [245, 215, 190, 165, 140, 115], "TE": [205, 175, 150, 125, 100, 80]}.items():
        for rank, score in enumerate(scores, 1):
            rows.append({"player_id": f"{position}{rank}", "player": f"{position} {rank}", "team": "X", "position": position, "current_adp": len(rows) + 1, "reference_points": score, "league_points": score})
    return pd.DataFrame(rows)


def _run(players, roster=REFERENCE_ROSTER):
    return estimate_adjusted_adp(players, "reference_points", "league_points", REFERENCE_ROSTER, roster, 2, 2)


def test_identical_scoring_preserves_market_order():
    result, _ = _run(_players())
    assert result.sort_values("league_adjusted_rank")["player_id"].tolist() == _players().sort_values("current_adp")["player_id"].tolist()


def test_uniform_additive_position_points_does_not_create_artificial_boost():
    players = _players()
    players.loc[players.position == "WR", "league_points"] += 20
    result, _ = _run(players)
    assert result.set_index("player_id").loc["WR1", "league_adjusted_rank"] == result.set_index("player_id").loc["WR1", "current_adp_rank"]


def test_elite_separation_moves_elite_player_up():
    players = _players()
    players.loc[players.player_id == "TE1", "league_points"] += 100
    result, _ = _run(players)
    te1 = result.set_index("player_id").loc["TE1"]
    assert te1.league_adjusted_rank < te1.current_adp_rank


def test_more_required_starters_increases_position_value():
    players = _players()
    result, _ = _run(players, ("QB", "RB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX"))
    rb1 = result.set_index("player_id").loc["RB1"]
    assert rb1.league_scarcity_value > rb1.reference_scarcity_value


def test_global_point_scale_does_not_move_equivalent_players():
    players = _players()
    players["league_points"] *= 1.5
    result, _ = _run(players)
    assert result.sort_values("league_adjusted_rank")["player_id"].tolist() == _players().sort_values("current_adp")["player_id"].tolist()


def test_global_rerank_has_unique_coherent_adps():
    players = _players()
    players.loc[players.player_id == "QB1", "league_points"] += 80
    result, _ = _run(players)
    assert result.league_adjusted_rank.is_unique
    assert result.league_adjusted_adp.is_unique
    assert result.league_adjusted_adp.is_monotonic_increasing


def test_zero_projection_players_have_zero_value_below_replacement_in_every_environment():
    players = pd.concat(
        [_players(), pd.DataFrame([{"player_id": "retired_qb", "player": "Retired QB", "team": "FA", "position": "QB", "current_adp": 2, "reference_points": 0, "league_points": 0}])],
        ignore_index=True,
    )
    result, _ = _run(players, ("QB", "QB", "QB", "RB", "RB", "WR", "WR", "TE", "FLEX"))
    retired = result.set_index("player_id").loc["retired_qb"]
    assert retired.reference_value_above_replacement == 0
    assert retired.league_value_above_replacement == 0
    assert retired.scarcity_delta == 0
    assert not retired.has_usable_projection
    assert retired.league_adjusted_rank > result.loc[result.has_usable_projection, "league_adjusted_rank"].max()
