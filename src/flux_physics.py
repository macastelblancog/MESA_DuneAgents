"""
flux_physics.py
Geometría de dunas y distribución de flujo de arena.

Implementa:
    • Ecuaciones de volumen y ancho (Ecs. 2/2⁻¹, Robson & Baas 2023/2024)
    • Ancho de cuernos (Ec. 1)
    • Tasa de migración (Ec. 7, Elbelrhiti et al. 2008)
    • Distribución de flujo de arena barlovento → sotavento
    • Modos de outflux: 'Hersen' (saturado) y 'Duran' (escalado con influx)

Convención de coordenadas:
    Viento primario en −y. La proyección de ancho es perpendicular al viento.
    Para viento en −y: ancho proyectado = ancho lateral (x).
    Para viento oblicuo: proyección = W · |cos(θ_desvio)| = W · |sin(wind_angle_rad)|

    donde wind_angle_rad es el ángulo absoluto devuelto por get_angle() (≈3π/2 para [0,−1]).
    Se verifica: |sin(3π/2)| = 1 para viento primario ✓

Referencias:
    Robson & Baas (2023), GRL — Ecs. 1–5
    Robson & Baas (2024), ESD — Ec. 3 modo Duran §2.2
    Elbelrhiti et al. (2008) — c=45, W₀=16.6 m, qsat=79 m²/yr
"""

from __future__ import annotations
import math
import numpy as np
from .wind_regimes import get_angle


# ── Ecuaciones de volumen y geometría ─────────────────────────────────────────

def flank_volume(w: float, lambda2: float, lambda3: float) -> float:
    """Volumen de un flanco (Ec. 2 del paper).

    V = λ₂ · λ₃ · W³ / 6

    Parámetros
    ----------
    w       : ancho del flanco [m]
    lambda2 : ratio longitud cuerno / ancho flanco (adim)
    lambda3 : ratio altura / ancho (adim); paper 2024 = 1/3, código v1 = 1/6

    Retorna
    -------
    float : volumen [m³]
    """
    return lambda2 * lambda3 * (w ** 3) / 6.0


def width_from_volume(v: float, lambda2: float, lambda3: float) -> float:
    """Ancho equivalente de flanco desde volumen (inversa Ec. 2).

    W = (6V / (λ₂ · λ₃))^(1/3)

    Retorna 0.0 si v ≤ 0.
    """
    if v <= 0.0:
        return 0.0
    return (6.0 * v / (lambda2 * lambda3)) ** (1.0 / 3.0)


def horn_width(w: float, alpha: float, delta: float) -> float:
    """Ancho del cuerno de un flanco (Ec. 1).

    H = α · W + Δ/2

    Parámetros
    ----------
    w     : ancho del flanco [m]
    alpha : coeficiente (0.05)
    delta : offset [m] (4.6)
    """
    return alpha * w + delta / 2.0


def migration_rate(lw: float, rw: float, qsat: float, dt: float,
                   w0: float = 16.6, c: float = 45.0) -> float:
    """Distancia de migración por paso (Ec. 7).

    v_mig = c · qsat / (lw + rw + W₀)
    d_mig = v_mig · dt

    Parámetros
    ----------
    lw, rw : anchos de flanco izquierdo y derecho [m]
    qsat   : flujo saturado [m²/año]
    dt     : paso de tiempo [años]
    w0     : ancho de referencia [m] (16.6 m, Elbelrhiti 2008)
    c      : factor de velocidad (45, Elbelrhiti 2008)

    Retorna
    -------
    float : desplazamiento [m] por paso de tiempo dt
    """
    return c * qsat * dt / (lw + rw + w0)


def projected_width(w: float, wind_angle_rad: float) -> float:
    """Ancho proyectado perpendicularmente a la dirección del viento.

    Para viento en −y (angle = 3π/2): |sin(3π/2)| = 1 → proyección = W (correcto).
    Para viento lateral puro (angle = 0 o π): proyección = 0 (sin flujo absorbido).

    Parámetros
    ----------
    w              : ancho del flanco [m]
    wind_angle_rad : ángulo del viento en radianes (de get_angle())
    """
    return w * abs(math.sin(wind_angle_rad))


# ── Distribución de flujo ─────────────────────────────────────────────────────

def distribute_flux(
    agents: list,
    wind_vec: tuple[float, float],
    qsat: float,
    q0: float,
    dt: float,
    w0: float = 16.6,
    c: float = 45.0,
    alpha: float = 0.05,
    delta: float = 4.6,
    outflux_mode: str = 'Hersen',
    a_duran: float = 0.45,
    b_duran: float = 1.0,
) -> None:
    """Distribuye flujo de arena a todos los agentes para el paso actual.

    Modifica en-lugar los atributos _influx_l, _influx_r, _outflux_l,
    _outflux_r y _migration_vec de cada DuneAgent.

    Algoritmo (bbox simplificado):
        1. Ordena agentes barlovento → sotavento (mayor proyección sobre viento primero).
        2. Para cada agente (en orden):
            a. Calcula ancho proyectado de cada flanco.
            b. Asigna influx proporcional a proyección.
            c. Calcula outflux según outflux_mode.
            d. Propaga flux de cuernos a agentes sotavento que solapen lateralmente.
            e. Calcula vector de migración.

    Parámetros
    ----------
    agents       : lista de DuneAgent activos
    wind_vec     : (wx, wy) vector unitario de dirección del viento
    qsat, q0     : flujo saturado y ambiental [m²/año]
    dt           : paso de tiempo [años]
    w0, c        : parámetros de migración (Elbelrhiti 2008)
    alpha, delta : geometría del cuerno
    outflux_mode : 'Hersen' (qout = qsat) o 'Duran' (qout escalado con qin)
    a_duran, b_duran : parámetros Durán et al. (2011), a=0.45, b=1.0
    """
    if not agents:
        return

    wx, wy = wind_vec
    wind_angle = get_angle(wind_vec)

    # ── 1. Ordenar agentes: barlovento primero ─────────────────────────────────
    # Proyectar posición sobre la dirección del viento (mayor → más barlovento)
    # Para viento [0,−1]: proyección = x·0 + y·(−1) = −y → orden ascendente en −y
    #                                                     = descendente en y ✓
    def upwind_projection(agent) -> float:
        x, y = agent.pos
        return -(x * wx + y * wy)   # negativo porque queremos mayor → primero

    sorted_agents = sorted(agents, key=upwind_projection)

    # Flujo ambiental disponible (m²/año); se va consumiendo por las dunas de barlovento
    # Simplificación bbox: todos comparten el mismo flujo inicial q0
    # En Shapely se restaría el polígono absorbido — aquí usamos columnas independientes
    q_residual_by_x: dict[str, float] = {}   # clave = bin_x simplificado

    # ── 2. Procesar cada agente en orden ──────────────────────────────────────
    for agent in sorted_agents:
        x, y = agent.pos
        lw = agent.lw
        rw = agent.rw

        if lw <= 0.0 and rw <= 0.0:
            continue

        # Proyecciones de ancho (perpendicular al viento)
        proj_l = projected_width(lw, wind_angle)
        proj_r = projected_width(rw, wind_angle)

        # Flujo ambiental disponible en la columna de esta duna
        # Aproximación: usar q0 como flujo base (una duna no sombrea a las de barlovento)
        q_avail = q0  # m²/año — flujo libre

        # Influx por flanco (volumen por paso)
        influx_l = q_avail * proj_l * dt  # m²/yr · m · yr = m³
        influx_r = q_avail * proj_r * dt

        # Outflux por cuerno
        hl = horn_width(lw, alpha, delta)
        hr = horn_width(rw, alpha, delta)

        if outflux_mode == 'Hersen':
            outflux_l = qsat * hl * dt   # m²/yr · m · yr = m³
            outflux_r = qsat * hr * dt

        elif outflux_mode == 'Duran':
            # Durán et al. (2011): qout = (a·qin + b·qsat) · (W/H)
            # → outflux = (a·q_avail + b·qsat) · proj_l · dt
            q_eff_l = a_duran * q_avail + b_duran * qsat
            q_eff_r = a_duran * q_avail + b_duran * qsat
            outflux_l = q_eff_l * proj_l * dt
            outflux_r = q_eff_r * proj_r * dt

        else:
            raise ValueError(f"outflux_mode desconocido: '{outflux_mode}'. "
                             f"Usar 'Hersen' o 'Duran'.")

        # Asignar flujos al agente
        agent.receive_flux(influx_l, influx_r)
        agent.schedule_outflux(outflux_l, outflux_r)

        # ── Propagar flujo de cuernos a agentes sotavento ─────────────────────
        # El cuerno izquierdo (lw) emite en dirección +x del cuerno
        # El cuerno derecho (rw) emite en dirección −x del cuerno
        horn_flux_l = qsat * hl * dt  # volumen emitido por cuerno izq.
        horn_flux_r = qsat * hr * dt  # volumen emitido por cuerno der.

        # Posición de los cuernos (tip)
        x_horn_l = x - lw   # borde izquierdo
        x_horn_r = x + rw   # borde derecho

        for other in sorted_agents:
            # Solo dunas que están más sotavento (menor proyección upwind)
            if upwind_projection(other) <= upwind_projection(agent):
                continue
            ox, oy = other.pos
            olw, orw = other.lw, other.rw
            # Rango lateral del agente objetivo
            o_left = ox - olw
            o_right = ox + orw
            # Si el cuerno izquierdo del agente solapa lateralmente con el otro
            if o_left <= x_horn_l <= o_right:
                # Determina si solapa con flanco izq o der del objetivo
                if x_horn_l <= ox:   # izquierda del centroide → flanco izq del otro
                    other.receive_flux(horn_flux_l * 0.5, 0.0)
                else:
                    other.receive_flux(0.0, horn_flux_l * 0.5)
            if o_left <= x_horn_r <= o_right:
                if x_horn_r <= ox:
                    other.receive_flux(horn_flux_r * 0.5, 0.0)
                else:
                    other.receive_flux(0.0, horn_flux_r * 0.5)

        # ── Vector de migración ────────────────────────────────────────────────
        d = migration_rate(lw, rw, qsat, dt, w0, c)
        agent.set_migration((d * wx, d * wy))


def w_min_theoretical(alpha, delta):
    return (delta/2) / (1 - alpha)