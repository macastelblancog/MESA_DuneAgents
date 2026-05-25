"""
collision_rules.py
Detección y resolución de colisiones entre DuneAgents.

Implementa:
    possible_candidates() — pre-filtro euclidiano O(N²) con early exit (v3)
    centroid_intersect()  — detección real por centroide ↔ footprint
    collision_pairs()     — lista de pares colisionantes
    one_rule()            — resolución de colisión conservando volumen y COM
    classify_collision()  — tipo: 'merging' | 'exchange' | 'fragmentation'

Convención de colisión (Robson & Baas 2023, §2.4):
    El agente más pequeño comprueba si su centroide toca el footprint del mayor.
    Resultado: 1 a 3 dunas de salida (más calveo posible en el paso siguiente).

Tipos de colisión (paper 2024, Fig. 3):
    merging       : 1 duna de salida (N_out = 1) — reduce el número de dunas
    exchange      : 2 dunas de salida (N_out = 2) — conserva el número
    fragmentation : 3+ dunas de salida (N_out ≥ 3) — aumenta el número
"""

from __future__ import annotations
import math
import numpy as np
from .gamma_threshold import gamma_c
from .flux_physics import flank_volume, width_from_volume, horn_width


# ── Pre-filtro euclidiano (del código v3) ──────────────────────────────────────

def possible_candidates(
    positions: list[tuple[float, float]],
    lws: list[float],
    rws: list[float],
    lambda2: float,
) -> list[tuple[int, int]]:
    """Pre-filtro O(N²) por distancia euclidiana antes del test de centroide.

    Dos dunas son candidatas si su distancia euclidiana es menor que la mayor
    longitud posible del campo (lambda2 * max_total_width). Esto evita los
    tests Shapely costosos para pares lejanos.

    Parámetros
    ----------
    positions : lista de (x, y) para cada agente
    lws, rws  : anchos de flanco izquierdo y derecho
    lambda2   : ratio longitud / ancho (para calcular distancia máxima)

    Retorna
    -------
    list de (i, j) con i < j: pares candidatos
    """
    n = len(positions)
    if n < 2:
        return []

    max_w = max(lw + rw for lw, rw in zip(lws, rws))
    max_dist = lambda2 * max_w

    candidates = []
    for i in range(n):
        xi, yi = positions[i]
        for j in range(i + 1, n):
            xj, yj = positions[j]
            dist = math.hypot(xi - xj, yi - yj)
            if dist < max_dist:
                candidates.append((i, j))
    return candidates


# ── Detección de colisión ─────────────────────────────────────────────────────

def _bbox(x: float, y: float, lw: float, rw: float,
          lambda2: float) -> tuple[float, float, float, float]:
    """Bounding box (x_min, x_max, y_min, y_max) de la duna."""
    return (x - lw, x + rw, y - lambda2 * (lw + rw), y)


def centroid_intersect(
    x1: float, y1: float, lw1: float, rw1: float,
    x2: float, y2: float, lw2: float, rw2: float,
    lambda2: float,
) -> bool:
    """Comprueba si el centroide de la duna más pequeña toca el bbox de la mayor.

    El centroide de una duna simétrica está aproximadamente en (x, y - L_body/2).
    Para simplificar usamos el centroide del footprint rectangular.

    Retorna True si hay colisión.
    """
    w1 = lw1 + rw1
    w2 = lw2 + rw2

    # La duna más pequeña prueba su centroide contra la más grande
    if w1 <= w2:
        cx = x1
        cy = y1 - lambda2 * w1 / 2.0
        xmin, xmax, ymin, ymax = _bbox(x2, y2, lw2, rw2, lambda2)
    else:
        cx = x2
        cy = y2 - lambda2 * w2 / 2.0
        xmin, xmax, ymin, ymax = _bbox(x1, y1, lw1, rw1, lambda2)

    return xmin <= cx <= xmax and ymin <= cy <= ymax


def collision_pairs(
    agents: list,
    lambda2: float,
) -> list[tuple[int, int]]:
    """Lista de pares de agentes que están colisionando.

    Usa pre-filtro euclidiano (v3) + test de centroide.

    Parámetros
    ----------
    agents  : lista de DuneAgent
    lambda2 : ratio longitud / ancho del campo (valor medio del modelo)

    Retorna
    -------
    lista de (i, j) con i < j: índices de pares colisionantes
    """
    positions = [a.pos for a in agents]
    lws = [a.lw for a in agents]
    rws = [a.rw for a in agents]

    candidates = possible_candidates(positions, lws, rws, lambda2)
    pairs = []

    for i, j in candidates:
        x1, y1 = positions[i]
        x2, y2 = positions[j]
        if centroid_intersect(x1, y1, lws[i], rws[i],
                              x2, y2, lws[j], rws[j], lambda2):
            pairs.append((i, j))

    return pairs


# ── Resolución de colisión ────────────────────────────────────────────────────

def one_rule(
    x1: float, y1: float, lw1: float, rw1: float,
    x2: float, y2: float, lw2: float, rw2: float,
    lambda1: float, lambda2: float, lambda3: float,
    alpha: float, delta: float, qshift_ratio: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Resolución de colisión conservando volumen total y COM.

    Implementa la OneRule del código v3 (Robson & Baas 2024).

    Retorna
    -------
    (xs, ys, lws, rws) : listas de posiciones y anchos de las dunas de salida.
        1 elemento → merging
        2 elementos → exchange
        3+ elementos → fragmentation
    """
    # Volúmenes de los cuatro flancos
    v_l1 = flank_volume(lw1, lambda2, lambda3)
    v_r1 = flank_volume(rw1, lambda2, lambda3)
    v_l2 = flank_volume(lw2, lambda2, lambda3)
    v_r2 = flank_volume(rw2, lambda2, lambda3)

    vtot = v_l1 + v_r1 + v_l2 + v_r2

    # COM total ponderado por volumen
    # Centroide approx: izq en (x - lw/2), der en (x + rw/2)
    com_x1 = (x1 - lw1 / 2.0) * (v_l1 + v_r1) / (v_l1 + v_r1 + 1e-30) \
             if v_l1 + v_r1 > 0 else x1
    # Centroide sencillo del par
    com_x_total = (x1 * (v_l1 + v_r1) + x2 * (v_l2 + v_r2)) / (vtot + 1e-30)
    com_y_total = (y1 * (v_l1 + v_r1) + y2 * (v_l2 + v_r2)) / (vtot + 1e-30)

    # Determinar flancos que colisionan (bboxes solapados)
    # Simplificado: fusionar todos los flancos que se tocan
    # Por posición: si x1 < x2, lw1 toca rw2... etc. Usamos volumen total.
    v_merged = vtot
    w_merged = width_from_volume(v_merged, lambda2, lambda3)   # ancho total equiv.

    # Calcular umbral de asimetría máxima
    w_half = w_merged / 2.0
    gc = gamma_c(w_half, alpha, delta, lambda1, lambda2, qshift_ratio)

    # Fracción izquierda / derecha: por posición de los flancos absorbidos
    # Flancos con x ≤ com_x → lado izquierdo
    contributions = [
        (x1 - lw1 / 2.0, v_l1),   # centroide flanco izq duna 1
        (x1 + rw1 / 2.0, v_r1),   # centroide flanco der duna 1
        (x2 - lw2 / 2.0, v_l2),
        (x2 + rw2 / 2.0, v_r2),
    ]
    v_left = sum(v for cx, v in contributions if cx <= com_x_total)
    v_right = vtot - v_left

    # Anchos equivalentes de los dos flancos fusionados
    new_lw = width_from_volume(v_left, lambda2, lambda3)
    new_rw = width_from_volume(v_right, lambda2, lambda3)

    # Comprobar si la asimetría del fusionado supera el umbral
    out_xs: list[float] = []
    out_ys: list[float] = []
    out_lws: list[float] = []
    out_rws: list[float] = []

    w_min_flank = max(new_lw, new_rw)
    w_min_check = min(new_lw, new_rw)

    if w_min_check <= 0.0 or (w_min_flank / (w_min_check + 1e-30)) <= gc:
        # Duna fusionada dentro del umbral → 1 producto (merging)
        out_xs.append(com_x_total)
        out_ys.append(com_y_total)
        out_lws.append(new_lw)
        out_rws.append(new_rw)
    else:
        # La duna fusionada es demasiado asimétrica → se calva inmediatamente
        # → 2 dunas simétricas (exchange-like)
        w_child_l = width_from_volume(v_left, lambda2, lambda3)
        w_child_r = width_from_volume(v_right, lambda2, lambda3)

        # Duna principal (flanco mayor fusionado, simétrica)
        w_main = (w_child_l + w_child_r) / 2.0
        out_xs.append(com_x_total)
        out_ys.append(com_y_total)
        out_lws.append(w_main)
        out_rws.append(w_main)

        # Duna secundaria (flanco menor, también simétrica, posicionada sotavento)
        w_sec = abs(w_child_l - w_child_r) / 2.0
        if w_sec > 0.0:
            offset = lambda2 * w_main
            out_xs.append(com_x_total)
            out_ys.append(com_y_total - offset)
            out_lws.append(w_sec)
            out_rws.append(w_sec)

    return out_xs, out_ys, out_lws, out_rws


def classify_collision(n_outputs: int) -> str:
    """Clasifica el tipo de colisión por número de productos.

    Parámetros
    ----------
    n_outputs : número de dunas producidas por one_rule()

    Retorna
    -------
    'merging' | 'exchange' | 'fragmentation'
    """
    if n_outputs == 1:
        return 'merging'
    elif n_outputs == 2:
        return 'exchange'
    else:
        return 'fragmentation'
