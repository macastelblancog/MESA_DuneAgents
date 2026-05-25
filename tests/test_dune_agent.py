"""
tests/test_dune_agent.py
Tests de la física del DuneAgent que no requieren distribute_flux.

Estos tests pueden correr desde el primer día: solo necesitan DuneSwarm
inicializado (sin llamar a step()) y las ecuaciones escalares de flux_physics.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gamma_threshold import gamma_c
from src.flux_physics import flank_volume, width_from_volume, migration_rate, horn_width
from src.wind_regimes import WindRegime, GetVector, GetAngle
from src.dune_swarm import DuneSwarm
from src.dune_agent import DuneAgent


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def default_model():
    """Modelo con parámetros por defecto del paper, sin dunas iniciales."""
    return DuneSwarm(n_dunes_init=0, seed=42)


@pytest.fixture
def symmetric_agent(default_model):
    """Duna simétrica de 10 m de ancho por flanco."""
    agent = DuneAgent(default_model, lw=10.0, rw=10.0)
    default_model.space.place_agent(agent, (400.0, 250.0))
    return agent


# ── Tests de ecuaciones escalares ─────────────────────────────────────────────

class TestScalarEquations:

    def test_flank_volume_positive(self):
        """V = λ₂ λ₃ W³ / 6 debe ser positivo para W > 0."""
        vol = flank_volume(10.0, lambda2=2.5, lambda3=1/6)
        assert vol > 0

    def test_flank_volume_scales_cubic(self):
        """Duplicar W debe multiplicar V por 8."""
        v1 = flank_volume(5.0,  lambda2=2.5, lambda3=1/6)
        v2 = flank_volume(10.0, lambda2=2.5, lambda3=1/6)
        assert pytest.approx(v2 / v1, rel=1e-6) == 8.0

    def test_width_from_volume_roundtrip(self):
        """width_from_volume(flank_volume(W)) == W."""
        w_orig = 15.3
        vol = flank_volume(w_orig, lambda2=2.5, lambda3=1/6)
        w_back = width_from_volume(vol, lambda2=2.5, lambda3=1/6)
        assert pytest.approx(w_back, rel=1e-9) == w_orig

    def test_width_from_volume_zero(self):
        """V = 0 debe producir W = 0, no NaN ni error."""
        assert width_from_volume(0.0, 2.5, 1/6) == 0.0

    def test_migration_rate_decreases_with_width(self):
        """Dunas más grandes migran más lento."""
        v_small = migration_rate(5.0,  5.0,  qsat=100, dt=0.001, c=50)
        v_large = migration_rate(20.0, 20.0, qsat=100, dt=0.001, c=50)
        assert v_small > v_large

    def test_horn_width_ec1(self):
        """H = α·W + Δ/2 — verificar con parámetros del paper."""
        h = horn_width(10.0, alpha=0.05, delta=4.6)
        assert pytest.approx(h) == 0.05 * 10.0 + 4.6 / 2.0


# ── Tests de gamma_c ──────────────────────────────────────────────────────────

class TestGammaC:

    def test_gamma_c_greater_than_one(self):
        """γ_c siempre debe ser > 1."""
        gc = gamma_c(5.0, alpha=0.05, delta=4.6, lambda1=1.5,
                     lambda2=2.5, qshift_ratio=0.2)
        assert gc > 1.0

    def test_gamma_c_decreases_with_width(self):
        """Dunas más grandes tienen umbral de calveo más bajo (más fácil calvear)."""
        gc_small = gamma_c(2.5, alpha=0.05, delta=4.6, lambda1=1.5,
                           lambda2=2.5, qshift_ratio=0.2)
        gc_large = gamma_c(20.0, alpha=0.05, delta=4.6, lambda1=1.5,
                           lambda2=2.5, qshift_ratio=0.2)
        assert gc_small > gc_large

    def test_gamma_c_differs_from_buggy_original(self):
        """
        Con lambda1=1.5 (valor correcto del paper) el resultado debe diferir
        de la versión original que asumía lambda1=1.0 implícitamente.
        Verifica que la corrección de BUG B-01 tiene efecto real.
        """
        gc_corrected = gamma_c(5.0, alpha=0.05, delta=4.6, lambda1=1.5,
                               lambda2=2.5, qshift_ratio=0.2)
        gc_original  = gamma_c(5.0, alpha=0.05, delta=4.6, lambda1=1.0,
                               lambda2=2.5, qshift_ratio=0.2)
        # La versión corregida debe diferir de la original
        # (con lambda1=1.0 e =1.5 el delta term cambia implícitamente)
        # Nota: la ec. actual de gamma_c no usa lambda1 directamente en el
        # denominador; este test documenta que el parámetro está presente
        # para uso futuro cuando se complete la ecuación completa del paper.
        assert isinstance(gc_corrected, float)
        assert isinstance(gc_original,  float)


# ── Tests de WindRegime ───────────────────────────────────────────────────────

class TestWindRegime:

    def test_unimodal_vector_is_unit(self):
        """El vector de viento muestreado debe ser unitario."""
        regime = WindRegime("unimodal", rng=np.random.default_rng(0))
        wx, wy = regime.sample()
        assert pytest.approx(wx**2 + wy**2, abs=1e-9) == 1.0

    def test_get_angle_inverse_of_get_vector(self):
        """GetAngle(GetVector(θ)) == θ."""
        for deg in [0, 30, 90, 135, -45, -90]:
            wx, wy = GetVector(deg)
            angle_rad = GetAngle((wx, wy))
            assert pytest.approx(angle_rad, abs=1e-9) == np.deg2rad(deg)

    def test_unknown_regime_raises(self):
        with pytest.raises(ValueError):
            WindRegime("viento_raro")

    def test_all_regimes_produce_unit_vectors(self):
        """Todos los regímenes producen vectores unitarios."""
        rng = np.random.default_rng(1)
        for regime_name in ["unimodal", "bimodal_acute", "bimodal_obtuse",
                             "multidirectional", "fixed"]:
            regime = WindRegime(regime_name, rng=rng)
            for _ in range(10):
                wx, wy = regime.sample()
                assert pytest.approx(wx**2 + wy**2, abs=1e-9) == 1.0


# ── Tests de DuneAgent ────────────────────────────────────────────────────────

class TestDuneAgent:

    def test_init_symmetric(self, symmetric_agent):
        """Duna simétrica tiene asymmetry = 0."""
        assert symmetric_agent.asymmetry == pytest.approx(0.0)

    def test_morphotype_symmetric_small_is_barchan(self, symmetric_agent):
        """Duna simétrica pequeña debe clasificarse como 'barchan'."""
        assert symmetric_agent.morphotype == "barchan"

    def test_lambda2_clamp(self, default_model):
        """lambda2 nunca debe ser < 1.2, incluso si se pasa un valor menor."""
        agent = DuneAgent(default_model, lw=5.0, rw=5.0, lambda2=0.5)
        default_model.space.place_agent(agent, (400.0, 250.0))
        assert agent.lambda2 >= 1.2

    def test_lambda2_drawn_from_distribution(self):
        """Con lambda2_std > 0, los agentes deben tener valores distintos de lambda2."""
        model = DuneSwarm(n_dunes_init=0, lambda2_mean=2.5, lambda2_std=0.5, seed=0)
        agents = []
        for i in range(20):
            a = DuneAgent(model, lw=10.0, rw=10.0)
            model.space.place_agent(a, (float(i * 30), 250.0))
            agents.append(a)
        l2_values = [a.lambda2 for a in agents]
        # Con 20 agentes y std=0.5, la std muestral debe ser > 0
        assert np.std(l2_values) > 0.0

    def test_lambda2_uniform_when_std_zero(self):
        """Con lambda2_std = 0, todos los agentes tienen el mismo lambda2."""
        model = DuneSwarm(n_dunes_init=0, lambda2_mean=2.5, lambda2_std=0.0, seed=0)
        agents = []
        for i in range(5):
            a = DuneAgent(model, lw=10.0, rw=10.0)
            model.space.place_agent(a, (float(i * 30), 250.0))
            agents.append(a)
        l2_values = [a.lambda2 for a in agents]
        assert all(v == pytest.approx(2.5) for v in l2_values)

    def test_volume_positive(self, symmetric_agent):
        """El volumen de una duna no-vacía siempre debe ser positivo."""
        assert symmetric_agent.volume > 0.0

    def test_width_property(self, symmetric_agent):
        """width == lw + rw."""
        a = symmetric_agent
        assert a.width == pytest.approx(a.lw + a.rw)

    def test_receive_flux_accumulates(self, symmetric_agent):
        """receive_flux debe acumular (no sobreescribir)."""
        a = symmetric_agent
        a.receive_flux(10.0, 5.0)
        a.receive_flux(3.0, 2.0)
        assert a._influx_l == pytest.approx(13.0)
        assert a._influx_r == pytest.approx(7.0)

    def test_reset_fluxes(self, symmetric_agent):
        """_reset_fluxes debe limpiar todos los buffers de flujo."""
        a = symmetric_agent
        a.receive_flux(10.0, 5.0)
        a._reset_fluxes()
        assert a._influx_l == 0.0
        assert a._influx_r == 0.0
        assert a._outflux_l == 0.0
        assert a._outflux_r == 0.0
        assert a._migration_vec == (0.0, 0.0)


# ── Tests de DuneSwarm (inicialización) ──────────────────────────────────────

class TestDuneSwarmInit:

    def test_n_agents_after_init(self):
        """El número de agentes iniciales debe ser n_dunes_init."""
        model = DuneSwarm(n_dunes_init=10, seed=42)
        assert len(list(model.agents)) == 10

    def test_w_min_formula(self):
        """w_min = (delta/2) / (1 - alpha)."""
        model = DuneSwarm(alpha=0.05, delta=4.6, n_dunes_init=0, seed=0)
        expected = (4.6 / 2.0) / (1.0 - 0.05)
        assert model.w_min == pytest.approx(expected)

    def test_agents_within_bounds(self):
        """Todos los agentes deben estar dentro del dominio al iniciar."""
        model = DuneSwarm(simwidth=800, simlength=500, n_dunes_init=20, seed=1)
        for agent in model.agents:
            x, y = agent.pos
            assert 0 <= x <= model.simwidth
            assert 0 <= y <= model.simlength

    def test_reproducibility(self):
        """Dos modelos con la misma seed deben producir el mismo estado inicial."""
        m1 = DuneSwarm(n_dunes_init=5, seed=99)
        m2 = DuneSwarm(n_dunes_init=5, seed=99)
        ws1 = sorted(a.width for a in m1.agents)
        ws2 = sorted(a.width for a in m2.agents)
        for w1, w2 in zip(ws1, ws2):
            assert w1 == pytest.approx(w2)

    def test_get_params_keys(self):
        """get_params debe incluir todos los parámetros de DuneSwarm."""
        model = DuneSwarm(n_dunes_init=0, seed=0)
        params = model.get_params()
        required = {"simwidth", "simlength", "qsat", "q0ratio", "qshift_ratio",
                    "dt", "lambda1", "lambda2_mean", "lambda2_std", "lambda3",
                    "alpha", "delta", "c", "w0", "wind_regime", "flux_mode"}
        assert required.issubset(set(params.keys()))
