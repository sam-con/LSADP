from src.scoring import score_projection, unsupported_scoring_rules


def test_generic_direct_stat_scoring_and_unknown_rules_are_exposed():
    result = score_projection({"pass_yd": 300, "rec": 4, "rec_yd": 50}, {"pass_yd": 0.04, "rec": 1, "rec_yd": 0.1, "pass_yd_300": 3})
    assert result.points == 21
    assert result.applied_rules["pass_yd"] == 12
    assert result.unsupported_rules == ["pass_yd_300"]
    assert unsupported_scoring_rules({"rec": 1, "unknown": -1}) == ["unknown"]
