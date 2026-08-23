import pandas as pd

from src.scarcity import build_scarcity_frame


def test_replacement_is_first_player_after_actual_starter_boundary():
    players = pd.DataFrame(
        [
            {"player_id": f"QB{rank}", "position": "QB", "current_adp": rank, "points": 400 - rank}
            for rank in range(1, 31)
        ]
    )
    _, summary = build_scarcity_frame(players, "points", ("QB", "QB", "QB"), 8)
    assert summary["QB"]["replacement_rank"] == 25
