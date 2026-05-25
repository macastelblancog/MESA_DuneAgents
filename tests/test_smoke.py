"""
test_smoke.py
Tests básicos de humo: verifican que los módulos se importan y corren
sin errores en condiciones mínimas. No verifican corrección física.

Ejecutar con: pytest tests/test_smoke.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


# ── Imports básicos ───────────────────────────────────────────────────────────

def test_imports():
    from src.gamma_threshold import gamma_c
    from src.wind_regimes    import make_wind_regime, get_vector, get_angle
    from src.flux_physics    import (horn_width, flank_volume, width_from_volume,
                                     migration_rate)
    from src.dune_agent      import DuneAgent
    from src.dune_swarm      import DuneSwarm
    from src.run_storage     import RunStorage


# ── gamma_threshold ───────────────────────────────────────────────────────────

def test_gamma_c_basic():
    from src.gamma_threshold import gamma_c
    gc = gamma_c(w_min=5.0, alpha=0.05, delta=4.6,
                 lambda1=1.5, lambda2=2.5, qshift_ratio=0.20)
    assert gc > 1.0, "γ_c debe ser > 1"


def test_gamma_c_degenerate():
    from src.gamma_threshold import gamma_c
    # Cuando denom ≤ 0 debe retornar inf
    gc = gamma_c(w_min=5.0, alpha=0.5, delta=4.6,
                 lambda1=1.5, lambda2=0.1, qshift_ratio=0.01)
    assert gc == float("inf")


# ── flux_physics (ecuaciones escalares) ──────────────────────────────────────

def test_volume_roundtrip():
    from src.flux_physics import flank_volume, width_from_volume
    w = 10.0
    l2, l3 = 2.5, 1/6
    v = flank_volume(w, l2, l3)
    w2 = width_from_volume(v, l2, l3)
    assert abs(w - w2) < 1e-9, "Ec. 2 y su inversa deben ser consistentes"


def test_migration_rate_positive():
    from src.flux_physics import migration_rate
    v = migration_rate(lw=10.0, rw=10.0, qsat=100.0, dt=0.001, c=50.0)
    assert v > 0.0


def test_migration_rate_zero_width():
    from src.flux_physics import migration_rate
    v = migration_rate(lw=0.0, rw=0.0, qsat=100.0, dt=0.001, c=50.0)
    assert v == 0.0


# ── wind_regimes ──────────────────────────────────────────────────────────────

def test_wind_regimes_all_valid():
    from src.wind_regimes import make_wind_regime
    rng = np.random.default_rng(42)
    for name in ["unimodal", "bimodal_acute", "bimodal_obtuse",
                 "multidirectional", "fixed"]:
        regime = make_wind_regime(name)
        vec = regime.sample(rng)
        assert len(vec) == 2
        norm = (vec[0]**2 + vec[1]**2) ** 0.5
        assert abs(norm - 1.0) < 1e-9, f"Vector del régimen {name} no es unitario"


def test_wind_regime_unknown():
    from src.wind_regimes import make_wind_regime
    with pytest.raises(ValueError):
        make_wind_regime("nonexistent_regime")


# ── DuneAgent básico ──────────────────────────────────────────────────────────

def test_dune_agent_properties():
    from src.dune_swarm import DuneSwarm
    model = DuneSwarm(n_dunes_init=0, seed=1)
    from src.dune_agent import DuneAgent
    agent = DuneAgent(model, lw=10.0, rw=10.0)
    model.space.place_agent(agent, (100.0, 100.0))

    assert agent.width == 20.0
    assert agent.asymmetry == 0.0
    assert agent.morphotype == "barchan"


def test_dune_agent_lambda2_fixed():
    from src.dune_swarm import DuneSwarm
    from src.dune_agent import DuneAgent
    model = DuneSwarm(n_dunes_init=0, lambda2_std=0.0, seed=1)
    agent = DuneAgent(model, 10.0, 10.0, lambda2=3.0)
    assert agent.lambda2 == 3.0


def test_dune_agent_lambda2_stochastic():
    from src.dune_swarm import DuneSwarm
    from src.dune_agent import DuneAgent
    model = DuneSwarm(n_dunes_init=0, lambda2_mean=2.5, lambda2_std=0.5, seed=99)
    lambdas = [DuneAgent(model, 10.0, 10.0).lambda2 for _ in range(50)]
    # Con std=0.5 debe haber variación
    assert max(lambdas) - min(lambdas) > 0.1, "lambda2 debe variar cuando lambda2_std > 0"
    assert all(l >= 1.2 for l in lambdas), "lambda2 debe ser siempre ≥ 1.2"


# ── DuneSwarm — smoke test ────────────────────────────────────────────────────

def test_swarm_init():
    from src.dune_swarm import DuneSwarm
    model = DuneSwarm(n_dunes_init=5, seed=42)
    assert len(list(model.agents)) == 5


def test_swarm_step_runs():
    from src.dune_swarm import DuneSwarm
    model = DuneSwarm(n_dunes_init=5, inject=False, seed=42)
    for _ in range(3):
        model.step()
    assert model.current_step == 3


def test_swarm_datacollector():
    from src.dune_swarm import DuneSwarm
    model = DuneSwarm(n_dunes_init=5, inject=False, seed=42)
    for _ in range(5):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    assert len(df) == 5
    assert "N_dunes" in df.columns
    assert "mean_width" in df.columns


def test_swarm_params_roundtrip():
    from src.dune_swarm import DuneSwarm
    params = dict(n_dunes_init=3, qsat=80.0, lambda2_std=0.3, seed=7)
    model = DuneSwarm(**params)
    p = model.get_params()
    assert p["qsat"] == 80.0
    assert p["lambda2_std"] == 0.3
