import numpy as np
import pandas as pd

from src.adp_model import _calibrate_position_curves, _curve_coordinates, _smooth_position_curve_deltas, estimate_adjusted_adp


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


def test_roster_changes_do_not_reshuffle_players_within_a_position():
    players = _players()
    result, _ = estimate_adjusted_adp(
        players,
        "reference_points",
        "league_points",
        REFERENCE_ROSTER,
        ("QB", "RB", "RB", "WR", "WR", "TE", "FLEX"),
        2,
        2,
    )
    assert result["raw_equivalent_rank_nudge"].eq(0).all()
    assert result["adjusted_market_pos_rank"].eq(result["market_pos_rank"]).all()


def test_elite_separation_moves_elite_player_up():
    players = _players()
    players.loc[players.player_id == "TE1", "league_points"] += 500
    result, _ = _run(players)
    te1 = result.set_index("player_id").loc["TE1"]
    assert te1.curve_strength_delta > 0
    assert te1.draft_score > te1.market_strength


def test_player_specific_scoring_residual_can_change_market_slot_within_position():
    players = _players()
    players.loc[players.player_id == "TE6", "league_points"] += 500
    result, _ = _run(players)
    te6 = result.set_index("player_id").loc["TE6"]
    assert te6.equivalent_rank_nudge > 0
    assert te6.adjusted_market_pos_rank < te6.market_pos_rank


def test_more_required_starters_increases_position_value():
    players = _players()
    result, _ = estimate_adjusted_adp(
        players,
        "reference_points",
        "league_points",
        ("QB", "RB", "WR", "TE"),
        ("QB", "RB", "RB", "RB", "WR", "WR", "TE"),
        1,
        2,
    )
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
    assert retired.equivalent_rank_nudge <= 0


def test_tied_replacement_vorp_slots_keep_their_own_reference_market_strength():
    """A long zero-VORP tail must not collapse to one averaged ADP strength."""
    ranks = np.arange(1, 21)
    reference_values = np.concatenate(([1.0], np.zeros(19)))
    frame = pd.DataFrame(
        {
            "position": "TE",
            "market_pos_rank": ranks,
            "market_strength": np.linspace(1.0, 0.2, 20),
            "reference_curve_coordinate": reference_values - (ranks - 1) * 1e-9,
            "league_curve_coordinate": np.concatenate(([1.0], np.linspace(0.20, 0.01, 19))) - (ranks - 1) * 1e-9,
            "reference_scarcity_value": reference_values,
            "league_scarcity_value": np.concatenate(([1.0], np.linspace(0.20, 0.01, 19))),
        }
    )
    calibrated = _calibrate_position_curves(frame)
    assert np.allclose(calibrated["reference_position_curve_strength"], frame["market_strength"])
    # The second TE can move toward a better *TE market slot*, but it cannot
    # inherit the strength averaged across the entire replacement-level tail.
    assert calibrated.loc[1, "position_curve_delta"] < 0.1


def test_tied_curve_coordinates_restart_at_each_tied_value():
    coordinates = _curve_coordinates(np.array([0.5, 0.0, 0.0, 0.0, -0.2, -0.2]))
    assert coordinates[1] == 0.0
    assert coordinates[1] > coordinates[2] > coordinates[3] > coordinates[4]
    assert coordinates[4] == -0.2
    assert coordinates[4] > coordinates[5]


def test_position_curve_smoothing_spreads_a_single_slot_cliff_without_changing_position_mean():
    frame = pd.DataFrame(
        {
            "position": ["RB"] * 9,
            "market_pos_rank": np.arange(1, 10),
            "market_strength": np.linspace(1.0, 0.2, 9),
            "position_curve_delta": [0.0, 0.0, 0.0, 0.0, -0.30, -0.30, -0.30, -0.30, -0.30],
        }
    )
    smoothed = _smooth_position_curve_deltas(frame)
    raw = frame["position_curve_delta"].to_numpy()
    adjusted = smoothed["position_curve_delta"].to_numpy()

    assert np.isclose(adjusted.mean(), raw.mean())
    assert np.abs(np.diff(adjusted)).max() < np.abs(np.diff(raw)).max()
    assert smoothed.loc[3, "position_curve_delta"] < 0
    assert smoothed.loc[4, "position_curve_delta"] > -0.30
    final_strength = smoothed["market_strength"] + smoothed["position_curve_delta"]
    assert final_strength.is_monotonic_decreasing
