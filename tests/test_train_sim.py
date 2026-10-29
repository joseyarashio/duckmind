from duckmind.train_sim import run_twin_experiment


def test_thirty_day_twins_show_caused_preference_not_random_personality_drift() -> None:
    first = run_twin_experiment(days=30)
    second = run_twin_experiment(days=30)

    assert first == second
    assert first["duck_a"]["final_preference"] > first["duck_b"]["final_preference"]
    assert first["duck_a"]["engaged_days"] > first["duck_b"]["engaged_days"]
    assert first["duck_b"]["final_preference"] == 0.5
    assert first["duck_a"]["personality_sociability"] == 0.5
    assert first["duck_b"]["personality_sociability"] == 0.5
    assert first["ablation"]["final_preference"] == first["duck_b"]["final_preference"]
    assert first["causal_preference_delta"] > 0


def test_year_long_twin_learning_stays_bounded_and_ablatable() -> None:
    result = run_twin_experiment(days=365)

    assert result["days"] == 365
    assert 0.5 < result["duck_a"]["final_preference"] < 1.0
    assert result["duck_a"]["learning_updates"] <= 365
    assert result["duck_a"]["personality_sociability"] == 0.5
    assert result["duck_b"]["final_preference"] == 0.5
    assert result["ablation"]["final_preference"] == result["duck_b"]["final_preference"]
    assert result["duck_a"]["engaged_days"] > result["duck_b"]["engaged_days"]
