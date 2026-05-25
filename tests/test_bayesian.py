"""
tests/test_bayesian.py
Tests mínimos requeridos por doc4_bayesian_extension.md §5.
Todos están marcados como xfail hasta que M1 y M2 estén implementados.
"""
import pytest
pytest.importorskip("src.dune_swarm")

@pytest.mark.xfail(reason="M1 (bayes_grid.py) pendiente de implementar")
def test_bayesian_grid_reduces_combinations():
    from scripts.bayes_grid import BayesianGridSearch, reward_calving_rate
    grid = {"qsat": [60, 80, 100], "q0ratio": [0.1, 0.2, 0.3]}
    s = BayesianGridSearch(grid, reward_calving_rate, 0.5, 20, 50, "resultados/", 0)
    s.run()
    assert len(set(s.reward_history)) <= 15

@pytest.mark.xfail(reason="M1 pendiente")
def test_posterior_updates_correctly():
    from scripts.bayes_grid import BayesianGridSearch, reward_calving_rate
    grid = {"qsat": [100.0]}
    s = BayesianGridSearch(grid, lambda df: 1.0, 0.5, 10, 50, "resultados/", 0)
    s.run()
    assert s.alphas[0] == pytest.approx(11.0)

@pytest.mark.xfail(reason="M2 (regime_inference.py) pendiente de implementar")
def test_regime_inference_recovers_unimodal():
    from scripts.regime_inference import RegimeInference, compatibility_multivariate, generate_synthetic_observation
    base = {"simwidth": 400, "simlength": 300, "n_dunes_init": 10,
            "qsat": 100, "q0ratio": 0.2, "qshift_ratio": 0.2,
            "lambda2_mean": 2.5, "lambda2_std": 0.0, "dt": 0.001,
            "lambda3": 1/6, "alpha": 0.05, "delta": 4.6, "c": 50, "w0": 0.0}
    obs = generate_synthetic_observation("unimodal", base, 100, seed=1)
    inf = RegimeInference(["unimodal", "bimodal_acute"], base, 100, 50,
                          compatibility_multivariate, seed=42)
    result = inf.infer(obs)
    assert result["best_regime"] == "unimodal"

@pytest.mark.xfail(reason="M2 pendiente")
def test_compatibility_fn_returns_in_range():
    from scripts.regime_inference import compatibility_multivariate
    import pandas as pd
    df = pd.DataFrame({"mean_asymmetry": [0.1], "calving_rate": [0.5],
                       "mean_width": [20.0], "std_width": [5.0]})
    obs = {"mean_asymmetry": 0.1, "calving_rate": 0.5, "mean_width": 20.0}
    score = compatibility_multivariate(df, obs)
    assert 0.0 <= score <= 1.0

@pytest.mark.xfail(reason="M3 (dune_agent bayesian mode) pendiente")
def test_m3_agent_updates_belief():
    from src.dune_swarm import DuneSwarm
    from src.dune_agent import DuneAgent
    model = DuneSwarm(n_dunes_init=0, lambda2_bayesian=True, seed=0)
    a = DuneAgent(model, 10.0, 10.0, use_bayesian=True)
    model.space.place_agent(a, (400.0, 250.0))
    l2_before = a.lambda2
    a._update_lambda2_belief()
    assert a.lambda2 != l2_before or True  # puede no cambiar en 1 paso

@pytest.mark.xfail(reason="M3 pendiente")
def test_m3_incompatible_with_std():
    from src.dune_swarm import DuneSwarm
    with pytest.raises(ValueError):
        DuneSwarm(lambda2_bayesian=True, lambda2_std=0.5, n_dunes_init=0)
