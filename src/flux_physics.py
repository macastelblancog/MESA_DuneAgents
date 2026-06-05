"""
flux_physics.py
Geometría de dunas y distribución de flujo de arena.

Implementa:
    • Ecuaciones de volumen y ancho (Ecs. 2/2⁻¹, Robson & Baas 2023/2024)
    • Ancho de cuernos (Ec. 1)
    • Tasa de migración (Ec. 7, Elbelrhiti et al. 2008)
    • Distribución de flujo de arena barlovento → sotavento
    • Modos de outflux: 'Hersen' (saturado) y 'Duran' (escalado con influx)

Algoritmo de distribute_flux (fiel al paper Robson & Baas 2024):
    Usa Shapely para calcular sombras y solapamientos exactos.
    El campo de flujo ambiental (fluxfield) se reduce con la sombra de cada duna
    procesada barlovento→sotavento. El flujo de cuernos también se propaga
    usando polígonos Shapely, conservando masa exactamente.

    Correcciones respecto a versión anterior:
        F-01: sombra de flujo — cada duna reduce el fluxfield para las sotavento
        F-02: influx = overlap_width * q * dt (no q * proj * dt)
        F-03: outflux Duran: qout = (a*qin_rate + b*qsat) * lw/H * H * dt
              donde qin_rate = leftinflux / (lw * dt)
        F-04: propagación de cuernos sin factor 0.5, usando overlap geométrico real
        F-05: cap de outflux: no puede superar el volumen disponible

Referencias:
    Robson & Baas (2023), GRL — Ecs. 1–5
    Robson & Baas (2024), ESD — Ec. 3 modo Duran §2.2, IterationCalculations
    Elbelrhiti et al. (2008) — c=45, W₀=16.6 m, qsat=79 m²/yr
"""

from __future__ import annotations
import math
import numpy as np
from .wind_regimes import get_angle

try:
    from shapely.geometry import box as shp_box, Polygon, MultiPolygon, Point
    from shapely.ops import unary_union
    from shapely.affinity import rotate as shp_rotate
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


# ── Ecuaciones de volumen y geometría ─────────────────────────────────────────

def flank_volume(w: float, lambda2: float, lambda3: float) -> float:
    """Volumen de un flanco (Ec. 2 del paper).

    V = λ₂ · λ₃ · W³ / 6
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
    """
    return alpha * w + delta / 2.0


def migration_rate(lw: float, rw: float, qsat: float, dt: float,
                   w0: float = 16.6, c: float = 45.0) -> float:
    """Distancia de migración por paso (Ec. 7).

    v_mig = c · qsat / (lw + rw + W₀)
    d_mig = v_mig · dt
    """
    return c * qsat * dt / (lw + rw + w0)


def projected_width(w: float, wind_angle_rad: float) -> float:
    """Ancho proyectado perpendicularmente a la dirección del viento."""
    return w * abs(math.sin(wind_angle_rad))


# ── Helpers Shapely ───────────────────────────────────────────────────────────

def _rotate_poly(poly, wind_vec: tuple[float, float]):
    """Rota un polígono para alinearlo con el viento (igual que paper Rotate())."""
    if not _SHAPELY:
        return poly
    wx, wy = wind_vec
    wv3 = np.array([wx, wy, 0.0])
    primary = np.array([0.0, -1.0, 0.0])
    cross = np.cross(wv3, primary)
    theta = float(np.arcsin(cross[-1]))
    if np.sign(wy) == 1:
        theta = math.pi - theta
    theta_deg = math.degrees(theta)
    return shp_rotate(poly, theta_deg, origin=(0, 0), use_radians=False)


def _barchan_poly(x: float, y: float, lw: float, rw: float,
                  lambda1: float, lambda2: float,
                  alpha: float, delta: float):
    """Crea polígonos Shapely del flanco izquierdo y derecho de la duna."""
    if not _SHAPELY or lw <= 0 or rw <= 0:
        # Fallback: bbox rectangular
        if _SHAPELY:
            left  = shp_box(x - lw, y - lambda2 * lw, x, y)
            right = shp_box(x, y - lambda2 * rw, x + rw, y)
            return left, right
        return None, None

    bodylength = lambda1 * (lw + rw) / 2.0

    # Flanco izquierdo
    lhw = alpha * lw + delta / 2.0
    pl = [(x, y),
          (x - lw, y - bodylength),
          (x - lw + lhw / 2.0, y - lambda2 * lw),
          (x, y - bodylength)]
    try:
        left = Polygon(pl)
        if not left.is_valid:
            left = shp_box(x - lw, y - lambda2 * lw, x, y)
    except Exception:
        left = shp_box(x - lw, y - lambda2 * lw, x, y)

    # Flanco derecho
    rhw = alpha * rw + delta / 2.0
    pr = [(x, y),
          (x + rw, y - bodylength),
          (x + rw - rhw / 2.0, y - lambda2 * rw),
          (x, y - bodylength)]
    try:
        right = Polygon(pr)
        if not right.is_valid:
            right = shp_box(x, y - lambda2 * rw, x + rw, y)
    except Exception:
        right = shp_box(x, y - lambda2 * rw, x + rw, y)

    return left, right


def _shadow(fluxfield, left, right, wind_vec):
    """Resta la sombra de la duna del campo de flujo (paper Shadow())."""
    if not _SHAPELY or left is None or right is None:
        return fluxfield
    try:
        barchan = unary_union([left, right])
        if not fluxfield.intersects(barchan):
            return fluxfield

        # Obtener coordenadas del barchan
        if isinstance(barchan, Polygon):
            xb = list(barchan.exterior.coords.xy[0])
            yb = list(barchan.exterior.coords.xy[1])
        else:
            xb, yb = [], []
            for geom in barchan.geoms:
                xb += list(geom.exterior.coords.xy[0])
                yb += list(geom.exterior.coords.xy[1])

        # Extender hacia sotavento para crear sombra
        field_bounds = fluxfield.bounds
        miny = field_bounds[1]

        shadow_pts = [(xi + wind_vec[0] * 1e-4, yi + wind_vec[1] * 1e-4)
                      for xi, yi in zip(xb, yb)]
        shadow_pts.append((min(xb), miny))
        shadow_pts.append((max(xb), miny))

        from shapely.geometry import MultiPoint
        shadow = MultiPoint(shadow_pts).convex_hull
        return fluxfield.difference(shadow)
    except Exception:
        return fluxfield


def _overlap_width(poly1, poly2) -> float:
    """Ancho de solapamiento entre dos polígonos (paper OverlapWidth())."""
    if not _SHAPELY or poly1 is None or poly2 is None:
        return 0.0
    try:
        if not poly1.intersects(poly2):
            return 0.0
        overlap = poly1.intersection(poly2)
        if overlap.is_empty:
            return 0.0
        bounds = overlap.bounds
        if len(bounds) == 0:
            return 0.0
        return abs(bounds[2] - bounds[0])
    except Exception:
        return 0.0


def _horn_poly(hx: float, hy: float, hw: float, miny: float):
    """Polígono rectangular del cuerno (paper HornPoly())."""
    if not _SHAPELY:
        return None
    return shp_box(hx - hw / 2.0, miny, hx + hw / 2.0, hy - 1e-3)


# ── Distribución de flujo — implementación fiel al paper ─────────────────────

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
    lambda1: float = 1.0,
    lambda2: float = 1.8,
) -> None:
    """Distribuye flujo de arena a todos los agentes para el paso actual.

    Implementación fiel a IterationCalculations() del paper (Robson & Baas 2024).

    Con Shapely disponible:
        - El campo de flujo ambiental (fluxfield) se reduce con la sombra de
          cada duna procesada de barlovento a sotavento.
        - El influx se calcula como overlap_width * q * dt.
        - Los cuernos propagan su flujo usando polígonos rectangulares que
          intersecan con los flancos de dunas sotavento.
        - El outflux está capeado al volumen disponible (F-05).

    Sin Shapely: fallback al algoritmo de proyección simplificado.
    """
    if not agents:
        return

    wx, wy = wind_vec
    wind_angle = get_angle(wind_vec)

    # ── Ordenar barlovento → sotavento ────────────────────────────────────────
    def upwind_proj(agent) -> float:
        x, y = agent.pos
        return -(x * wx + y * wy)

    sorted_agents = sorted(agents, key=upwind_proj, reverse=True)

    # Cache de polígonos rotados por agente — evita recalcular para cada emisor
    _poly_cache: dict[int, tuple] = {}

    def _get_rotated_polys(ag):
        aid = id(ag)
        if aid not in _poly_cache:
            ax, ay = ag.pos
            alw, arw = ag.lw, ag.rw
            al2 = getattr(ag, 'lambda2', lambda2)
            lo, ro = _barchan_poly(ax, ay, alw, arw, lambda1, al2, alpha, delta)
            _poly_cache[aid] = (_rotate_poly(lo, wind_vec), _rotate_poly(ro, wind_vec))
        return _poly_cache[aid]

    if not _SHAPELY:
        # ── Fallback sin Shapely ───────────────────────────────────────────────
        for agent in sorted_agents:
            x, y   = agent.pos
            lw, rw = agent.lw, agent.rw
            if lw <= 0.0 and rw <= 0.0:
                continue

            proj_l = projected_width(lw, wind_angle)
            proj_r = projected_width(rw, wind_angle)
            influx_l = q0 * proj_l * dt
            influx_r = q0 * proj_r * dt

            hl = horn_width(lw, alpha, delta)
            hr = horn_width(rw, alpha, delta)

            lv = flank_volume(lw, lambda2, 1/3)
            rv = flank_volume(rw, lambda2, 1/3)

            if outflux_mode == 'Hersen':
                outflux_l = min(qsat * hl * dt, lv + influx_l)
                outflux_r = min(qsat * hr * dt, rv + influx_r)
            else:
                qin_rate_l = influx_l / (lw * dt) if lw > 0 else 0
                qin_rate_r = influx_r / (rw * dt) if rw > 0 else 0
                qout_l = (a_duran * qin_rate_l + b_duran * qsat) * lw / hl
                qout_r = (a_duran * qin_rate_r + b_duran * qsat) * rw / hr
                outflux_l = min(qout_l * hl * dt, lv + influx_l)
                outflux_r = min(qout_r * hr * dt, rv + influx_r)

            agent.receive_flux(influx_l, influx_r)
            agent.schedule_outflux(outflux_l, outflux_r)

            d = migration_rate(lw, rw, qsat, dt, w0, c)
            agent.set_migration((d * wx, d * wy))
        return

    # ── Implementación con Shapely (fiel al paper) ────────────────────────────

    # Campo de flujo inicial: rectángulo que cubre todo el dominio
    # Usamos un bbox muy grande — se rotará con el viento
    BIG = 1e7
    fluxfield = shp_box(-BIG, -BIG, BIG, BIG)

    # Rotar el campo al sistema de referencia del viento
    fluxfield = _rotate_poly(fluxfield, wind_vec)
    try:
        miny = fluxfield.bounds[1]
    except Exception:
        miny = -BIG

    for agent in sorted_agents:
        x, y   = agent.pos
        lw, rw = agent.lw, agent.rw
        if lw <= 0.0 and rw <= 0.0:
            continue

        # Volúmenes actuales de cada flanco
        l2 = getattr(agent, 'lambda2', lambda2)
        l3 = agent.model.lambda3
        lv = flank_volume(lw, l2, l3)
        rv = flank_volume(rw, l2, l3)

        # Geometría de polígonos en sistema rotado con el viento
        left_orig, right_orig = _barchan_poly(x, y, lw, rw, lambda1, l2, alpha, delta)
        left_rot  = _rotate_poly(left_orig,  wind_vec)
        right_rot = _rotate_poly(right_orig, wind_vec)

        # F-01: calcular influx como solapamiento con fluxfield actual
        # F-02: restar sombra del campo antes de medir solapamiento
        fluxfield_after = _shadow(fluxfield, left_rot, right_rot, wind_vec)

        leftinflux  = _overlap_width(fluxfield_after, left_rot)  * q0 * dt
        rightinflux = _overlap_width(fluxfield_after, right_rot) * q0 * dt

        # Actualizar campo de flujo (sombra acumulada)
        fluxfield = fluxfield_after

        # Geometría del cuerno en sistema rotado
        hl = horn_width(lw, alpha, delta)
        hr = horn_width(rw, alpha, delta)

        lhx_orig = x - lw + hl / 2.0
        lhy_orig = y - l2 * lw
        rhx_orig = x + rw - hr / 2.0
        rhy_orig = y - l2 * rw

        # Rotar posiciones de cuernos
        def _rot_xy(px, py):
            wv3 = np.array([wx, wy, 0.0])
            cross = np.cross(wv3, np.array([0., -1., 0.]))
            theta = float(np.arcsin(cross[-1]))
            if np.sign(wy) == 1:
                theta = math.pi - theta
            mat = np.array([[math.cos(theta), -math.sin(theta)],
                            [math.sin(theta),  math.cos(theta)]])
            return mat.dot([px, py])

        lhx, lhy = _rot_xy(lhx_orig, lhy_orig)
        rhx, rhy = _rot_xy(rhx_orig, rhy_orig)
        lhx = float(np.nan_to_num(lhx))
        lhy = float(np.nan_to_num(lhy))
        rhx = float(np.nan_to_num(rhx))
        rhy = float(np.nan_to_num(rhy))

        # F-03: outflux según modo
        if outflux_mode == 'Hersen':
            qout_l = qsat
            qout_r = qsat
        else:
            # Duran: qout = (a * qin_rate + b * qsat) * lw / H
            qin_rate_l = leftinflux  / (lw * dt) if lw > 0 else 0.0
            qin_rate_r = rightinflux / (rw * dt) if rw > 0 else 0.0
            qout_l = (a_duran * qin_rate_l + b_duran * qsat) * lw / hl
            qout_r = (a_duran * qin_rate_r + b_duran * qsat) * rw / hr

        # F-05: cap — outflux no puede superar volumen disponible
        outflux_l = min(qout_l * hl * dt, lv + leftinflux)
        outflux_r = min(qout_r * hr * dt, rv + rightinflux)

        agent.receive_flux(leftinflux, rightinflux)
        agent.schedule_outflux(outflux_l, outflux_r)

        # F-04: propagar flujo de cuernos a dunas sotavento
        lefthorn  = _horn_poly(lhx, lhy, hl, miny)
        righthorn = _horn_poly(rhx, rhy, hr, miny)

        # Radio máximo de alcance del cuerno — pre-filtro euclidiano
        l2 = getattr(agent, 'lambda2', lambda2)
        horn_reach = l2 * max(lw, rw) * 2.0

        for other in sorted_agents:
            if other is agent:
                continue
            if upwind_proj(other) >= upwind_proj(agent):
                continue

            ox, oy = other.pos

            # Pre-filtro euclidiano — descartar pares lejanos sin Shapely
            dist = math.hypot(ox - x, oy - y)
            if dist > horn_reach:
                continue

            olw = other.lw
            orw = other.rw
            ol2 = getattr(other, 'lambda2', lambda2)

            other_left_rot, other_right_rot = _get_rotated_polys(other)

            if lefthorn is not None:
                ll = _overlap_width(lefthorn, other_left_rot)  * qout_l * dt
                lr = _overlap_width(lefthorn, other_right_rot) * qout_l * dt
                if ll + lr > 0:
                    other.receive_flux(ll, lr)

            if righthorn is not None:
                rl = _overlap_width(righthorn, other_left_rot)  * qout_r * dt
                rr = _overlap_width(righthorn, other_right_rot) * qout_r * dt
                if rl + rr > 0:
                    other.receive_flux(rl, rr)

        # Vector de migración
        d = migration_rate(lw, rw, qsat, dt, w0, c)
        agent.set_migration((d * wx, d * wy))


def w_min_theoretical(alpha, delta):
    return (delta / 2) / (1 - alpha)