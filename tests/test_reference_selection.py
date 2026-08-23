from src.models import select_reference_league


def test_one_qb_uses_matching_standard_half_and_ppr_markets():
    roster = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
    assert select_reference_league({"rec": 0}, roster).adp_field == "adp_std"
    assert select_reference_league({"rec": 0.5}, roster).adp_field == "adp_half_ppr"
    assert select_reference_league({"rec": 1}, roster).adp_field == "adp_ppr"


def test_superflex_and_two_qb_use_available_two_qb_market():
    assert select_reference_league({"rec": 0.5}, ["QB", "RB", "SUPER_FLEX"]).adp_field == "adp_2qb"
    assert select_reference_league({"rec": 1}, ["QB", "QB", "RB"]).adp_field == "adp_2qb"


def test_custom_reception_scoring_selects_nearest_conventional_market():
    roster = ["QB", "RB", "WR", "TE"]
    assert select_reference_league({"rec": 0.3}, roster).adp_field == "adp_half_ppr"
    assert select_reference_league({"rec": 1.5}, roster).adp_field == "adp_ppr"
