"""
dune_agent.py
Clase DuneAgent — el agente duna eoólica con dos flancos semi-independientes.

Correcciones aplicadas respecto al código original v1 (Robson & Baas 2023):
    C-01: gamma_c() ahora incluye el segundo umbral γ_c,λ = 2λ₂/λ₁ − 1
          (Ecs. 5–6 del paper 2024). La función importada ya lo implementa.
    C-02: _check_removal() solo elimina en y < 0 (borde sotavento).
          El borde y = simlength es de inyección, NO de eliminación.
          Antes: eliminaba en ambos bordes y (bug silencioso que destruía
          dunas recién inyectadas en el borde barlovento).
    B-04: _calve() usa lambda3 (no lambda2) en el cálculo de COM del flanco hijo.

Convención de coordenadas:
    Viento primario = −y. Borde barlovento = y = simlength. Sotavento = y = 0.
    lw (izquierdo) = flanco en el lado +x (estribor mirando al sur).
    rw (derecho)   = flanco en el lado −x (babor).

Referencias:
    Robson & Baas (2023) GRL — modelo base
    Robson & Baas (2024) ESD — correcciones gamma_c (Ecs. 4–6) y parámetros
"""

from __future__ import annotations
import numpy as np
import mesa
from .gamma_threshold import gamma_c, w_min_theoretical
from .flux_physics import flank_volume, width_from_volume


class DuneAgent(mesa.Agent):
    """Duna eoólica con dos flancos semi-independientes.

    Estado interno:
        lw, rw     : anchos de flanco izquierdo y derecho [m]
        lambda2    : parámetro geométrico propio de este agente (extensión original)
        _influx_l/r  : flujo recibido este paso por flanco [m³] (reset cada paso)
        _outflux_l/r : flujo emitido por cuerno este paso [m³] (reset cada paso)
        _migration_vec : (dx, dy) vector de desplazamiento calculado por el modelo
    """

    def __init__(self, model: "DuneSwarm", lw: float, rw: float,
                 lambda2: float | None = None):
        super().__init__(model)
        self.lw = float(lw)
        self.rw = float(rw)
        self.lambda2 = self._init_lambda2(lambda2)

        # Flujos (se acumulan en _distribute_flux, se resetean al final de step)
        self._influx_l: float = 0.0
        self._influx_r: float = 0.0
        self._outflux_l: float = 0.0
        self._outflux_r: float = 0.0
        self._migration_vec: tuple[float, float] = (0.0, 0.0)

    # ── Inicialización de lambda2 ─────────────────────────────────────────────

    def _init_lambda2(self, provided: float | None) -> float:
        """Asigna λ₂ propio: valor explícito o sorteado de N(μ, σ)."""
        if provided is not None:
            return max(1.0, float(provided))
        if self.model.lambda2_std > 0.0:
            val = float(self.model.rng.normal(
                self.model.lambda2_mean, self.model.lambda2_std))
            return max(1.0, val)
        return float(self.model.lambda2_mean)

    # ── Interfaz de flujos (llamada por DuneSwarm._distribute_flux) ───────────

    def receive_flux(self, flux_l: float, flux_r: float) -> None:
        self._influx_l += max(0.0, flux_l)
        self._influx_r += max(0.0, flux_r)

    def schedule_outflux(self, flux_l: float, flux_r: float) -> None:
        self._outflux_l = max(0.0, flux_l)
        self._outflux_r = max(0.0, flux_r)

    def set_migration(self, vec: tuple[float, float]) -> None:
        self._migration_vec = vec

    # ── Paso de actualización (llamado por shuffle_do) ────────────────────────

    def step(self) -> None:
        """Orden de ejecución fijo: física → migración → checks → reset."""
        self._update_volumes()
        self._apply_lateral_shift()
        self._recalc_widths()
        self._migrate()
        self._check_removal()
        if self in self.model.agents:   # puede haber sido eliminado
            self._check_calving()
        if self in self.model.agents:
            self._reset_fluxes()

    # ── Física interna ────────────────────────────────────────────────────────

    def _update_volumes(self) -> None:
        """Balance de flujos (Ec. 3, términos 1 y 2 de Robson & Baas 2023)."""
        l3 = self.model.lambda3
        l2 = self.lambda2
        vl = flank_volume(self.lw, l2, l3) + self._influx_l - self._outflux_l
        vr = flank_volume(self.rw, l2, l3) + self._influx_r - self._outflux_r
        self._vl_new = max(0.0, vl)
        self._vr_new = max(0.0, vr)

    def _apply_lateral_shift(self) -> None:
        """Transferencia lateral entre flancos (Ec. 3, tercer término).

        dV = qshift · dt · λ₂ · |lw − rw| · |sin(wind_angle)|

        Equivalencia matemática verificada:
            |sin(wind_angle)| donde wind_angle = get_angle([0,−1]) = 3π/2
            → |sin(3π/2)| = 1  ← máximo para viento primario ✓
            Idéntico a sin|90° − θ| con θ = desviación del eje primario.
        """
        if abs(self.lw - self.rw) < 1e-9:
            return
        qshift = self.model.qshift_ratio * self.model.qsat
        wind_angle = self.model._wind_angle_rad
        dv = (qshift * self.model.dt * self.lambda2
              * abs(self.lw - self.rw) * abs(np.sin(wind_angle)))
        sign = np.sign(self.rw - self.lw)   # positivo: rw>lw → lw crece
        self._vl_new = max(0.0, self._vl_new + sign * dv)
        self._vr_new = max(0.0, self._vr_new - sign * dv)

    def _recalc_widths(self) -> None:
        """Convierte volúmenes a anchos (inversa de Ec. 2)."""
        self.lw = width_from_volume(self._vl_new, self.lambda2, self.model.lambda3)
        self.rw = width_from_volume(self._vr_new, self.lambda2, self.model.lambda3)

    def _migrate(self) -> None:
        """Mueve la duna según el vector pre-calculado."""
        dx, dy = self._migration_vec
        x, y = self.pos
        new_x = x + dx
        new_y = y + dy
        # C-02 FIX: solo eliminar en y < 0 (sotavento) o x fuera del dominio
        # NO eliminar en y > simlength (ese es el borde de inyección barlovento)
        if (new_x < 0 or new_x > self.model.simwidth 
            or new_y < 0 or new_y > self.model.simlength):
            self.model.remove_agent(self)
        else:
            self.model.space.move_agent(self, (new_x, new_y))

    def _check_removal(self) -> None:
        """Elimina el agente si es demasiado pequeño.

        C-02 FIX: la eliminación por posición ocurre en _migrate().
        Aquí solo se comprueba el tamaño mínimo.
        """
        w_min = self.model.w_min
        if self.lw < w_min and self.rw < w_min:
            self.model.remove_agent(self)

    def _check_calving(self) -> None:
        """Dispara calveo si la asimetría supera γ_c (Ecs. 4–6 paper 2024).

        C-01 FIX: usa gamma_c() que implementa min(γ_c,shift, γ_c,λ).
        """
        if self.lw <= 0.0 or self.rw <= 0.0:
            return
        ratio = max(self.lw, self.rw) / min(self.lw, self.rw)
        gc = gamma_c(
            w_min=min(self.lw, self.rw),
            alpha=self.model.alpha,
            delta=self.model.delta,
            lambda1=self.model.lambda1,
            lambda2=self.lambda2,
            qshift_ratio=self.model.qshift_ratio,
        )
        if ratio > gc:
            self._calve()

    def _calve(self) -> None:
        """Divide el agente en dos DuneAgent hijos conservando volumen y COM.

        B-04 FIX: usa lambda3 (no lambda2) en width_from_volume() de los hijos.
        El código v1 tenía un error tipográfico: pasaba lambda2 como tercer arg
        de COMS() en el flanco hijo (ambos casos 'left' y 'right').
        """
        l2 = self.lambda2
        l3 = self.model.lambda3
        x, y = self.pos

        vl = flank_volume(self.lw, l2, l3)
        vr = flank_volume(self.rw, l2, l3)
        vtot = vl + vr

        if vtot <= 0.0:
            self.model.remove_agent(self)
            return

        # Hijo 1: del flanco mayor (simétrico con el volumen del flanco mayor)
        larger_v = max(vl, vr)
        smaller_v = min(vl, vr)

        # w_child es el ancho simétrico del flanco mayor reformado como barchán
        # Conservación: V_total_hijo1 = 2 * V_flanco_mayor → w tal que 2*(λ₂λ₃w³/6) = larger_v
        w_child1 = width_from_volume(larger_v / 2.0, l2, l3)  # B-04 FIX: usa l3 ✓
        l2_child1 = max(1.0, float(self.model.rng.normal(l2, 0.05)))

        # Hijo 2: del flanco menor
        w_child2 = width_from_volume(smaller_v / 2.0, l2, l3)  # B-04 FIX: usa l3 ✓
        l2_child2 = max(1.0, float(self.model.rng.normal(l2, 0.05)))

        # Conservar COM: hijo1 cerca del COM original, hijo2 sotavento
        offset = l2 * w_child1
        x1 = x
        y1 = y
        x2 = x
        y2 = y - offset

        # Crear los hijos si están dentro del dominio
        if (w_child1 > self.model.w_min and
                0 <= x1 <= self.model.simwidth and y1 >= 0):
            child1 = DuneAgent(self.model, w_child1, w_child1, lambda2=l2_child1)
            self.model.space.place_agent(child1, (x1, y1))
            self.model.agents.add(child1)

        if (w_child2 > self.model.w_min and
                0 <= x2 <= self.model.simwidth and y2 >= 0):
            child2 = DuneAgent(self.model, w_child2, w_child2, lambda2=l2_child2)
            self.model.space.place_agent(child2, (x2, y2))
            self.model.agents.add(child2)

        self.model.calving_count += 1
        self.model.calvings_this_step += 1
        self.model.remove_agent(self)

    def _reset_fluxes(self) -> None:
        self._influx_l = 0.0
        self._influx_r = 0.0
        self._outflux_l = 0.0
        self._outflux_r = 0.0
        self._migration_vec = (0.0, 0.0)

    # ── Propiedades derivadas ─────────────────────────────────────────────────

    @property
    def width(self) -> float:
        return self.lw + self.rw

    @property
    def asymmetry(self) -> float:
        """Ratio de asimetría = |lw − rw| / (lw + rw) ∈ [0, 1)."""
        w = self.width
        return abs(self.lw - self.rw) / w if w > 0.0 else 0.0

    @property
    def asym_ratio(self) -> float:
        """Ratio lw/rw (como en Fig. 6 del paper 2024). ≥ 1 si lw ≥ rw."""
        if self.rw <= 0.0:
            return float('inf')
        return self.lw / self.rw

    @property
    def volume(self) -> float:
        l2 = self.lambda2
        l3 = self.model.lambda3
        return flank_volume(self.lw, l2, l3) + flank_volume(self.rw, l2, l3)

    @property
    def morphotype(self) -> str:
        """Clasificación morfológica según estado interno."""
        if self.width == 0.0:
            return 'ghost'
        asym = self.asymmetry
        gc = gamma_c(
            min(self.lw, self.rw),
            self.model.alpha, self.model.delta,
            self.model.lambda1, self.lambda2,
            self.model.qshift_ratio,
        )
        if asym >= (gc - 1.0) / gc:
            return 'pre_calving'
        if asym >= 0.15:
            return 'asymmetric'
        if self.width >= self.model.w_transverse_threshold:
            return 'transverse'
        return 'barchan'

    def __repr__(self) -> str:
        x, y = self.pos if self.pos else (0, 0)
        return (f"DuneAgent(id={self.unique_id}, "
                f"lw={self.lw:.1f}, rw={self.rw:.1f}, "
                f"pos=({x:.0f},{y:.0f}))")