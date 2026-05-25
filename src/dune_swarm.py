"""
dune_swarm.py
Clase DuneSwarm — modelo MESA que orquesta el campo de dunas barchán.

Funcionalidades:
    • Carga de parámetros desde JSON (DuneSwarm.from_json / from_params)
    • Modos de outflux: 'Hersen' (paper 2023) y 'Duran' (paper 2024)
    • Modos de inyección: 'wmin', 'weq' (paper 2024, Ec. 8), 'fixed'
    • Conteo de tipos de colisión: merging / exchange / fragmentation
    • Perfil longitudinal (binning 500 m) para reproducir Figs. 4/5 del paper 2024
    • DataCollector con métricas de modelo y agente

Correcciones integradas:
    C-01: gamma_c con segundo término γ_c,λ (via dune_agent que usa gamma_threshold)
    C-02: _check_removal solo en y<0 (via dune_agent._migrate)
    C-03: lambda3 = 1/3 (paper 2024) o 1/6 (código original) — configurable por JSON

Convención de coordenadas:
    Viento primario = −y. Barlovento = y = simlength. Sotavento = y = 0.
    lw (izquierdo) = flanco +x. rw (derecho) = flanco −x.
    ADVERTENCIA: pasar wind_vec=[0,−1] como dirección primaria.

Referencias:
    Robson & Baas (2023) GRL; Robson & Baas (2024) ESD §2.5 (Ec. 8 inyección)
"""

from __future__ import annotations
import math
import numpy as np
import mesa
from pathlib import Path
from typing import Any

from .dune_agent import DuneAgent
from .wind_regimes import WindRegime, get_angle
from .flux_physics import distribute_flux, w_min_theoretical, flank_volume, width_from_volume, migration_rate
from .collision_rules import collision_pairs, one_rule, classify_collision


class DuneSwarm(mesa.Model):
    """Modelo de enjambre de dunas barchán.

    Instanciar directamente:
        swarm = DuneSwarm(simwidth=600, simlength=600, ...)

    O desde JSON (recomendado):
        swarm = DuneSwarm.from_json('params/paper_2024_esd.json')
        swarm = DuneSwarm.from_json('params/paper_2024_esd.json', seed=42)
    """

    # ── Constructores ─────────────────────────────────────────────────────────

    def __init__(
        self,
        # Geometría
        lambda1: float = 1.0,
        lambda2_mean: float = 1.8,
        lambda2_std: float = 0.0,
        lambda3: float = 1.0 / 6.0,
        alpha: float = 0.05,
        delta: float = 4.6,
        # Flujo
        qsat: float = 79.0,
        q0ratio: float = 0.25,
        qshift_ratio: float = 0.10,
        outflux_mode: str = 'Hersen',
        a_duran: float = 0.45,
        b_duran: float = 1.0,
        # Migración
        c: float = 45.0,
        w0: float = 16.6,
        # Tiempo
        dt: float = 0.125,
        n_steps: int = 100,
        # Dominio
        simwidth: float = 600.0,
        simlength: float = 600.0,
        fieldwidth: float = 200.0,
        fieldlength: float = 200.0,
        # Inyección
        inject: bool = False,
        inject_mode: str = 'wmin',
        rho0: float = 0.0,
        w_inject_fixed: float | None = None,
        # Viento
        wind_regime: str = 'unimodal',
        wind_mean_deg: float = 270.0,
        wind_std_deg: float = 3.0,
        wind_secondary_deg: float | None = None,
        wind_secondary_std_deg: float | None = None,
        wind_secondary_weight: float = 0.25,
        # Modelo
        collisions: bool = True,
        seed: int | None = None,
        w_min: float | None = None,
        w_transverse_threshold: float = 60.0,
        # Condiciones iniciales
        n_dunes_init: int = 10,
        lws_init: list[float] | None = None,
        rws_init: list[float] | None = None,
        xs_init: list[float] | None = None,
        ys_init: list[float] | None = None,
    ):
        super().__init__(seed=seed)

        # ── Parámetros físicos ─────────────────────────────────────────────────
        self.lambda1 = float(lambda1)
        self.lambda2_mean = float(lambda2_mean)
        self.lambda2_std = float(lambda2_std)
        self.lambda3 = float(lambda3)
        self.alpha = float(alpha)
        self.delta = float(delta)

        self.qsat = float(qsat)
        self.q0ratio = float(q0ratio)
        self.q0 = self.q0ratio * self.qsat
        self.qshift_ratio = float(qshift_ratio)
        self.outflux_mode = outflux_mode
        self.a_duran = float(a_duran)
        self.b_duran = float(b_duran)

        self.c = float(c)
        self.w0 = float(w0)
        self.dt = float(dt)
        self.n_steps = int(n_steps)

        self.w_min = float(w_min) if w_min is not None else w_min_theoretical(alpha, delta)
        self.w_transverse_threshold = float(w_transverse_threshold)

        # ── Dominio ────────────────────────────────────────────────────────────
        self.simwidth = float(simwidth)
        self.simlength = float(simlength)
        self.fieldwidth = float(fieldwidth)
        self.fieldlength = float(fieldlength)
        self.field_xmin = (simwidth - fieldwidth) / 2.0
        self.field_xmax = self.field_xmin + fieldwidth
        self.field_ymin = simlength - fieldlength  # borde sotavento del campo
        # NOTA: borde barlovento siempre = simlength (inyección)

        # ── Espacio continuo ───────────────────────────────────────────────────
        self.space = mesa.spaces.ContinuousSpace(
            x_min=0.0, x_max=simwidth,
            y_min=-1.0,               # permite un paso extra antes de eliminar
            y_max=simlength + 1.0,
            torus=False,
        )

        # ── Inyección ──────────────────────────────────────────────────────────
        self.inject = inject
        self.inject_mode = inject_mode
        self.rho0 = float(rho0)
        self.w_inject_fixed = float(w_inject_fixed) if w_inject_fixed is not None else None
        self._w_eq = self._calc_weq()  # tamaño de equilibrio W_eq para modo 'weq'

        # ── Régimen de viento ──────────────────────────────────────────────────
        self._wind_regime = WindRegime(
            regime=wind_regime,
            mean_deg=wind_mean_deg,
            std_deg=wind_std_deg,
            secondary_deg=wind_secondary_deg,
            secondary_std_deg=wind_secondary_std_deg,
            secondary_weight=wind_secondary_weight,
            rng=self.rng,
        )
        self._wind_vec: tuple[float, float] = (0.0, -1.0)   # default primario
        self._wind_angle_rad: float = get_angle(self._wind_vec)

        # ── Colisiones ─────────────────────────────────────────────────────────
        self.collisions_enabled = bool(collisions)

        # ── Contadores ─────────────────────────────────────────────────────────
        self.current_step: int = 0
        self.calving_count: int = 0
        self.collision_count: int = 0
        self.calvings_this_step: int = 0
        self.collisions_this_step: int = 0
        self.merging_count: int = 0
        self.exchange_count: int = 0
        self.fragmentation_count: int = 0
        self.merging_this_step: int = 0
        self.exchange_this_step: int = 0
        self.fragmentation_this_step: int = 0

        # ── DataCollector ──────────────────────────────────────────────────────
        self.datacollector = mesa.DataCollector(
            model_reporters={
                'step':                   'current_step',
                'N_dunes':                lambda m: len(list(m.agents)),
                'mean_width':             lambda m: self._safe_mean(
                                              [a.width for a in m.agents]),
                'std_width':              lambda m: self._safe_std(
                                              [a.width for a in m.agents]),
                'mean_asymmetry':         lambda m: self._safe_mean(
                                              [a.asymmetry for a in m.agents]),
                'calvings_this_step':     'calvings_this_step',
                'collisions_this_step':   'collisions_this_step',
                'merging_this_step':      'merging_this_step',
                'exchange_this_step':     'exchange_this_step',
                'fragmentation_this_step':'fragmentation_this_step',
                'calving_count':          'calving_count',
                'collision_count':        'collision_count',
                'merging_count':          'merging_count',
                'exchange_count':         'exchange_count',
                'fragmentation_count':    'fragmentation_count',
                'wind_angle_deg':         lambda m: math.degrees(m._wind_angle_rad),
            },
            agent_reporters={
                'lw':         'lw',
                'rw':         'rw',
                'width':      'width',
                'asymmetry':  'asymmetry',
                'asym_ratio': 'asym_ratio',
                'lambda2':    'lambda2',
                'morphotype': 'morphotype',
                'pos_x':      lambda a: a.pos[0] if a.pos else None,
                'pos_y':      lambda a: a.pos[1] if a.pos else None,
            },
        )

        # ── Crear dunas iniciales ──────────────────────────────────────────────
        self._create_initial_dunes(
            n_dunes_init, lws_init, rws_init, xs_init, ys_init)

    # ── Constructores desde JSON ──────────────────────────────────────────────

    @classmethod
    def from_json(cls, json_path: str | Path, seed: int | None = None) -> "DuneSwarm":
        """Crea DuneSwarm desde un archivo JSON de parámetros.

        Uso:
            swarm = DuneSwarm.from_json('params/paper_2024_esd.json')
            swarm = DuneSwarm.from_json('params/paper_2023_grl.json', seed=42)
        """
        from .params_loader import load_params, params_to_swarm_kwargs, describe_params
        params = load_params(json_path)
        print(describe_params(params))
        kwargs = params_to_swarm_kwargs(params)
        if seed is not None:
            kwargs['seed'] = seed
        return cls(**kwargs)

    @classmethod
    def from_params(cls, params: dict, seed: int | None = None) -> "DuneSwarm":
        """Crea DuneSwarm desde un dict de parámetros (ya cargado)."""
        from .params_loader import params_to_swarm_kwargs
        kwargs = params_to_swarm_kwargs(params)
        if seed is not None:
            kwargs['seed'] = seed
        return cls(**kwargs)

    # ── Paso principal ────────────────────────────────────────────────────────

    def step(self) -> None:
        """Ejecuta un paso de simulación completo.

        Orden fijo (garantiza simultaneidad efectiva sin SimultaneousActivation):
        [1] Contadores → [2] Viento → [3] Flujos → [4] Agentes →
        [5] Colisiones → [6] Inyección → [7] DataCollector
        """
        # [1] Contadores
        self.current_step += 1
        self.calvings_this_step = 0
        self.collisions_this_step = 0
        self.merging_this_step = 0
        self.exchange_this_step = 0
        self.fragmentation_this_step = 0

        # [2] Sortear dirección de viento
        self._wind_vec = self._wind_regime.sample()
        self._wind_angle_rad = get_angle(self._wind_vec)

        # [3] Distribuir flujo de arena a todos los agentes
        active = list(self.agents)
        if active:
            distribute_flux(
                agents=active,
                wind_vec=self._wind_vec,
                qsat=self.qsat,
                q0=self.q0,
                dt=self.dt,
                w0=self.w0,
                c=self.c,
                alpha=self.alpha,
                delta=self.delta,
                outflux_mode=self.outflux_mode,
                a_duran=self.a_duran,
                b_duran=self.b_duran,
            )

        # [4] Actualizar todos los agentes (orden aleatorio para evitar sesgos)
        self.agents.shuffle_do('step')

        # [5] Detectar y resolver colisiones
        if self.collisions_enabled:
            self._resolve_collisions()

        # [6] Inyectar nuevas dunas (si está habilitado)
        if self.inject:
            self._inject_dunes()

        # [7] Registrar estado
        self.datacollector.collect(self)

    def run(self, n_steps: int | None = None) -> None:
        """Corre la simulación por n_steps pasos (o self.n_steps si no se especifica)."""
        n = n_steps if n_steps is not None else self.n_steps
        for i in range(n):
            self.step()
            if (i + 1) % max(1, n // 10) == 0:
                n_agents = len(list(self.agents))
                print(f"  Paso {i+1}/{n} | dunas: {n_agents} | "
                      f"calveos: {self.calving_count} | "
                      f"colisiones: {self.collision_count}")

    # ── Colisiones ────────────────────────────────────────────────────────────

    def _resolve_collisions(self) -> None:
        """Detecta y resuelve colisiones entre agentes activos."""
        active = list(self.agents)
        if len(active) < 2:
            return

        pairs = collision_pairs(active, self.lambda2_mean)
        # Orden aleatorio para evitar sesgos de resolución
        pair_order = list(range(len(pairs)))
        self.rng.shuffle(pair_order)

        for idx in pair_order:
            i, j = pairs[idx]
            # Verificar que ambos agentes siguen activos (pueden haberse
            # eliminado en una colisión anterior de este mismo paso)
            if i >= len(active) or j >= len(active):
                continue
            a1 = active[i]
            a2 = active[j]
            if a1 not in self.agents or a2 not in self.agents:
                continue

            x1, y1 = a1.pos
            x2, y2 = a2.pos

            xs, ys, lws, rws = one_rule(
                x1, y1, a1.lw, a1.rw,
                x2, y2, a2.lw, a2.rw,
                self.lambda1, self.lambda2_mean, self.lambda3,
                self.alpha, self.delta, self.qshift_ratio,
            )

            # Conteo de tipo
            ctype = classify_collision(len(xs))
            if ctype == 'merging':
                self.merging_count += 1
                self.merging_this_step += 1
            elif ctype == 'exchange':
                self.exchange_count += 1
                self.exchange_this_step += 1
            else:
                self.fragmentation_count += 1
                self.fragmentation_this_step += 1

            self.collision_count += 1
            self.collisions_this_step += 1

            # Eliminar las dos dunas originales
            self.remove_agent(a1)
            self.remove_agent(a2)

            # Crear productos
            for k in range(len(xs)):
                lw_k = lws[k]
                rw_k = rws[k]
                x_k = xs[k]
                y_k = ys[k]
                if (lw_k > self.w_min and rw_k > self.w_min
                        and 0 <= x_k <= self.simwidth and y_k >= 0):
                    child = DuneAgent(self, lw_k, rw_k)
                    self.space.place_agent(child, (x_k, y_k))
                    self.agents.add(child)

    # ── Inyección ─────────────────────────────────────────────────────────────

    def _inject_dunes(self) -> None:
        """Inyecta nuevas dunas en el borde barlovento (y = simlength).

        Modos:
            'wmin'  : ancho = w_min (mínimo viable)
            'weq'   : ancho = W_eq = Δ·qsat / (q₀ − α·qsat)  [paper 2024, Ec. 8]
            'fixed' : ancho = w_inject_fixed

        Tasa de inyección (Ec. 8 paper 2024):
            N_enter = ρ₀ · v_mig(W_eq) · fieldwidth · dt
        """
        wx, wy = self._wind_vec

        # Calcular ancho de inyección según modo
        if self.inject_mode == 'weq':
            w_inj = self._w_eq
        elif self.inject_mode == 'fixed' and self.w_inject_fixed is not None:
            w_inj = self.w_inject_fixed
        else:
            w_inj = self.w_min * 2.0  # 'wmin': duna mínima viable

        # Velocidad de migración de las dunas inyectadas
        v_mig = migration_rate(w_inj / 2.0, w_inj / 2.0, self.qsat, self.dt, self.w0, self.c)

        # Número esperado de dunas a inyectar este paso
        n_inject_float = self.rho0 * abs(v_mig) * self.fieldwidth
        n_inject = int(n_inject_float)
        # Parte fraccionaria: inyectar una duna extra con probabilidad proporcional
        if self.rng.uniform() < (n_inject_float - n_inject):
            n_inject += 1

        for _ in range(n_inject):
            # Posición: x uniforme en el campo central, y en el borde barlovento
            x = float(self.rng.uniform(self.field_xmin, self.field_xmax))
            # Inyectar ligeramente por debajo del borde para evitar C-02 falsos
            y = self.simlength - abs(float(self.rng.uniform(0.0, 0.001)))

            if 0 <= x <= self.simwidth and y >= 0:
                new_agent = DuneAgent(self, w_inj / 2.0, w_inj / 2.0)
                self.space.place_agent(new_agent, (x, y))
                self.agents.add(new_agent)

    # ── Creación de dunas iniciales ───────────────────────────────────────────

    def _create_initial_dunes(
        self,
        n: int,
        lws: list[float] | None,
        rws: list[float] | None,
        xs: list[float] | None,
        ys: list[float] | None,
    ) -> None:
        """Crea las dunas iniciales del campo."""
        # Usar listas explícitas si se proporcionan
        if lws and rws and xs and ys and len(lws) == len(rws) == len(xs) == len(ys):
            for lw, rw, x, y in zip(lws, rws, xs, ys):
                agent = DuneAgent(self, lw, rw)
                self.space.place_agent(agent, (float(x), float(y)))
                self.agents.add(agent)
        elif n > 0:
            # Generar posiciones aleatorias dentro del campo
            for _ in range(n):
                lw = float(self.rng.uniform(self.w_min * 3, self.w_min * 15))
                rw = float(self.rng.uniform(self.w_min * 3, self.w_min * 15))
                x = float(self.rng.uniform(self.field_xmin, self.field_xmax))
                y = float(self.rng.uniform(
                    self.field_ymin, self.simlength - 10.0))
                agent = DuneAgent(self, lw, rw)
                self.space.place_agent(agent, (x, y))
                self.agents.add(agent)

    # ── Análisis de perfil longitudinal ──────────────────────────────────────

    def longitudinal_profile(self, bin_size: float = 500.0) -> list[dict]:
        """Perfil de anchura media y densidad de dunas por bin downwind.

        Para reproducir Figs. 4 y 5 del paper 2024. Los bins van de
        y=simlength (barlovento) a y=0 (sotavento), en tramos de bin_size metros.

        Parámetros
        ----------
        bin_size : tamaño del bin en metros [m] (default 500 m como en el paper)

        Retorna
        -------
        lista de dicts con keys: bin_center, downwind_dist, mean_width, std_width,
        dune_density (dunas/m²), n_dunes
        """
        n_bins = int(self.simlength / bin_size)
        results = []

        for b in range(n_bins):
            y_lo = b * bin_size
            y_hi = (b + 1) * bin_size
            # "downwind distance" = distancia desde el borde barlovento
            downwind = self.simlength - y_hi

            dunes_in_bin = [
                a for a in self.agents
                if a.pos is not None and y_lo <= a.pos[1] < y_hi
            ]
            widths = [a.width for a in dunes_in_bin]

            n = len(dunes_in_bin)
            area = bin_size * self.fieldwidth
            results.append({
                'bin_center': (y_lo + y_hi) / 2.0,
                'downwind_dist': downwind,
                'n_dunes': n,
                'mean_width': float(np.mean(widths)) if widths else 0.0,
                'std_width': float(np.std(widths)) if len(widths) > 1 else 0.0,
                'dune_density': n / area if area > 0 else 0.0,
            })

        return results

    # ── Gestión de agentes ────────────────────────────────────────────────────

    def remove_agent(self, agent: DuneAgent) -> None:
        """Elimina un agente del modelo con limpieza completa."""
        if agent in self.agents:
            self.space.remove_agent(agent)
            self.agents.remove(agent)

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _calc_weq(self) -> float:
        """W_eq = Δ·qsat / (q₀ − α·qsat)  (tamaño de equilibrio del paper 2024)."""
        q0 = self.q0ratio * self.qsat
        denom = q0 - self.alpha * self.qsat
        if denom <= 0.0:
            return self.w_min * 5.0   # fallback
        return self.delta * self.qsat / denom

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _safe_std(values: list[float]) -> float:
        return float(np.std(values)) if len(values) > 1 else 0.0

    def __repr__(self) -> str:
        return (f"DuneSwarm(step={self.current_step}, "
                f"N={len(list(self.agents))}, "
                f"outflux={self.outflux_mode})")
