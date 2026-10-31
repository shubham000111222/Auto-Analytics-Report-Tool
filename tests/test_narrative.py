from engine.narrative import validate_narrative

STATS = {
    "n_rows": 1000, "n_cols": 12,
    "data_quality": {"worst_col": "age", "worst_pct": 23.5},
    "charts": {"numeric": [{"title": "Distribution of age", "facts": {"mean": 41.2, "skew": 1.8}}]},
}


def test_valid_narrative_passes():
    text = "- Age has the most missing data at 23.5%.\n- Average age is about 41.2 years."
    ok, unverified = validate_narrative(text, STATS)
    assert ok and unverified == []


def test_fabricated_number_fails():
    text = "- Revenue grew by 87% last quarter."
    ok, unverified = validate_narrative(text, STATS)
    assert not ok and "87" in unverified


def test_trivial_numbers_are_ignored():
    text = "- Here are the top 5 categories out of 10 reviewed."
    ok, unverified = validate_narrative(text, STATS)
    assert ok


def test_close_rounding_is_tolerated():
    text = "- Missing data in age is roughly 23%."
    ok, unverified = validate_narrative(text, STATS)
    assert ok
