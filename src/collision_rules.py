"""
collision_rules.py
Detección y resolución de colisiones entre DuneAgents.

Implementa:
    possible_candidates() — pre-filtro euclidiano O(N²) con early exit
    centroid_intersect()  — detección real por centroide ↔ footprint (bbox)
    collision_pairs()     — lista de pares colisionantes
    one_rule()            — resolución fiel al paper (Robson & Baas 2024)
    classify_collision()  — tipo: 'merging' | 'exchange' | 'fragmentation'

Correcciones respecto a versión anterior:
    C-01: one_rule ahora detecta qué flancos se intersectan con Shapely
    C-02: flancos no colisionantes se evalúan con gamma_c para decidir si
          se unen al fusionado o forman duna separada (fragmentation real)
    C-03: fracción izq/der ponderada por cvols**(1/3) como en el paper
    C-04: conservación del COM total de las dos dunas originales
    C-05: posición del producto corregida con término geométrico del paper

Convención de colisión (Robson & Baas 2023, §2.4):
    El agente más pequeño comprueba si su centroide toca el footprint del mayor.
    Resultado: 1 a 3 dunas de salida.

Tipos de colisión:
    merging       : 1 duna de salida — reduce el número
    exchange      : 2 dunas de salida — conserva el número
    fragmentation : 3+ dunas de salida — aumenta el número
"""

from __future__ import annotations
import math
import numpy as np
from .gamma_threshold import gamma_c
from .flux_physics import flank_volume, width_from_volume, horn_width

try:
    from shapely.geometry import Polygon, MultiPolygon, box as shp_box
    from shapely.ops import unary_union
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


# ── Helpers de volumen (equivalentes a VS del paper) ─────────────────────────

def _vs_total(w: float, otherw: float,
              lambda1: float, lambda2: float, lambda3: float) -> float:
    """Volumen total de un flanco (cuerpo + cuerno). Equivalente a VS(..., True)."""
    return lambda2 * lambda3 * w**3 / 6.0


def _equiv_width_flank(v: float, lambda2: float, lambda3: float) -> float:
    """Ancho equivalente de flanco desde volumen."""
    if v <= 0:
        return 0.0
    return (6.0 * v / (lambda2 * lambda3)) ** (1.0 / 3.0)


def _equiv_width_body(v: float, lambda2: float, lambda3: float) -> float:
    """Ancho equivalente de duna completa (simétrica) desde volumen total."""
    if v <= 0:
        return 0.0
    return 2.0 * (3.0 * v / (lambda2 * lambda3)) ** (1.0 / 3.0)


# ── Geometría Shapely ─────────────────────────────────────────────────────────

def _barchan_polys(x: float, y: float, lw: float, rw: float,
                   lambda1: float, lambda2: float,
                   alpha: float, delta: float):
    """Polígonos Shapely del flanco izquierdo y derecho."""
    if not _SHAPELY or lw <= 0 or rw <= 0:
        if _SHAPELY:
            left  = shp_box(x - lw, y - lambda2 * lw, x, y)
            right = shp_box(x, y - lambda2 * rw, x + rw, y)
            return left, right
        return None, None

    bodylength = lambda1 * (lw + rw) / 2.0
    lhw = alpha * lw + delta / 2.0
    rhw = alpha * rw + delta / 2.0

    try:
        left = Polygon([
            (x, y),
            (x - lw, y - bodylength),
            (x - lw + lhw / 2.0, y - lambda2 * lw),
            (x, y - bodylength),
        ])
        if not left.is_valid:
            left = shp_box(x - lw, y - lambda2 * lw, x, y)
    except Exception:
        left = shp_box(x - lw, y - lambda2 * lw, x, y)

    try:
        right = Polygon([
            (x, y),
            (x + rw, y - bodylength),
            (x + rw - rhw / 2.0, y - lambda2 * rw),
            (x, y - bodylength),
        ])
        if not right.is_valid:
            right = shp_box(x, y - lambda2 * rw, x + rw, y)
    except Exception:
        right = shp_box(x, y - lambda2 * rw, x + rw, y)

    return left, right


def _poly_width(poly) -> float:
    """Ancho (extensión en x) de un polígono Shapely."""
    if poly is None:
        return 0.0
    try:
        bounds = poly.bounds
        return abs(bounds[2] - bounds[0])
    except Exception:
        return 0.0


# ── Pre-filtro euclidiano ─────────────────────────────────────────────────────

def possible_candidates(
    positions: list[tuple[float, float]],
    lws: list[float],
    rws: list[float],
    lambda2: float,
) -> list[tuple[int, int]]:
    """Pre-filtro O(N²) por distancia euclidiana."""
    n = len(positions)
    if n < 2:
        return []

    max_w   = max(lw + rw for lw, rw in zip(lws, rws))
    max_dist = lambda2 * max_w

    candidates = []
    for i in range(n):
        xi, yi = positions[i]
        for j in range(i + 1, n):
            xj, yj = positions[j]
            if math.hypot(xi - xj, yi - yj) < max_dist:
                candidates.append((i, j))
    return candidates


# ── Detección de colisión ─────────────────────────────────────────────────────

def _bbox(x, y, lw, rw, lambda2):
    return (x - lw, x + rw, y - lambda2 * (lw + rw), y)


def centroid_intersect(
    x1, y1, lw1, rw1,
    x2, y2, lw2, rw2,
    lambda2: float,
) -> bool:
    """Comprueba si el centroide de la duna más pequeña toca el bbox de la mayor."""
    w1 = lw1 + rw1
    w2 = lw2 + rw2
    if w1 <= w2:
        cx = x1
        cy = y1 - lambda2 * w1 / 2.0
        xmin, xmax, ymin, ymax = _bbox(x2, y2, lw2, rw2, lambda2)
    else:
        cx = x2
        cy = y2 - lambda2 * w2 / 2.0
        xmin, xmax, ymin, ymax = _bbox(x1, y1, lw1, rw1, lambda2)
    return xmin <= cx <= xmax and ymin <= cy <= ymax


def collision_pairs(agents: list, lambda2: float) -> list[tuple[int, int]]:
    """Lista de pares de agentes colisionantes.

    Con Shapely: usa CentroidIntersect del paper (centroide real vs polígono real).
    Sin Shapely: usa bbox simplificado.
    """
    positions = [a.pos for a in agents]
    lws = [a.lw for a in agents]
    rws = [a.rw for a in agents]

    candidates = possible_candidates(positions, lws, rws, lambda2)
    pairs = []

    # Cache de polígonos para evitar recalcular
    poly_cache = {}
    def get_polys(idx):
        if idx not in poly_cache:
            model = agents[idx].model
            l1 = getattr(agents[idx], "lambda2", lambda2)
            poly_cache[idx] = _barchan_polys(
                positions[idx][0], positions[idx][1],
                lws[idx], rws[idx],
                model.lambda1, l1, model.alpha, model.delta)
        return poly_cache[idx]

    for i, j in candidates:
        if _SHAPELY:
            try:
                li, ri = get_polys(i)
                lj, rj = get_polys(j)
                # CentroidIntersect del paper: centroide de la menor vs polígono de la mayor
                wi = lws[i] + rws[i]
                wj = lws[j] + rws[j]
                from shapely.ops import unary_union
                if wi <= wj:
                    sl, sr = li, ri
                    large_union = unary_union([lj, rj])
                else:
                    sl, sr = lj, rj
                    large_union = unary_union([li, ri])
                # Tres centroides como en el paper
                cl = sl.centroid.buffer(0.01)
                cr_ = sr.centroid.buffer(0.01)
                cb = unary_union([sl, sr]).centroid.buffer(0.01)
                collide = (cl.intersects(large_union) or
                           cr_.intersects(large_union) or
                           cb.intersects(large_union))
            except Exception:
                # Fallback a bbox
                x1, y1 = positions[i]; x2, y2 = positions[j]
                collide = centroid_intersect(
                    x1, y1, lws[i], rws[i],
                    x2, y2, lws[j], rws[j], lambda2)
        else:
            x1, y1 = positions[i]; x2, y2 = positions[j]
            collide = centroid_intersect(
                x1, y1, lws[i], rws[i],
                x2, y2, lws[j], rws[j], lambda2)

        if collide:
            pairs.append((i, j))

    return pairs


# ── Resolución de colisión — fiel al paper ────────────────────────────────────

def one_rule(
    x1: float, y1: float, lw1: float, rw1: float,
    x2: float, y2: float, lw2: float, rw2: float,
    lambda1: float, lambda2: float, lambda3: float,
    alpha: float, delta: float, qshift_ratio: float,
) -> tuple[list, list, list, list]:
    """Resolución de colisión fiel a OneRule() del paper (Robson & Baas 2024).

    Algoritmo:
    1. Detectar qué flancos se intersectan con Shapely.
    2. Flancos colisionantes → fusionar en una duna.
    3. Flancos no colisionantes → evaluar con gamma_c:
       - Si compatible → unir a la fusionada
       - Si incompatible → duna separada (fragmentation)
    4. Ajustar posiciones para conservar COM total.

    Retorna (xs, ys, lws, rws) de las dunas de salida.
    """
    # Volúmenes de los cuatro flancos
    l1v = _vs_total(lw1, rw1, lambda1, lambda2, lambda3)
    r1v = _vs_total(rw1, lw1, lambda1, lambda2, lambda3)
    l2v = _vs_total(lw2, rw2, lambda1, lambda2, lambda3)
    r2v = _vs_total(rw2, lw2, lambda1, lambda2, lambda3)

    original_vtot = l1v + r1v + l2v + r2v
    if original_vtot <= 0:
        return [], [], [], []

    if _SHAPELY:
        # ── Detectar flancos colisionantes con Shapely ─────────────────────────
        left1,  right1 = _barchan_polys(x1, y1, lw1, rw1, lambda1, lambda2, alpha, delta)
        left2,  right2 = _barchan_polys(x2, y2, lw2, rw2, lambda1, lambda2, alpha, delta)

        try:
            barc1 = unary_union([left1, right1])
            barc2 = unary_union([left2, right2])
            comx1, comy1 = barc1.centroid.x, barc1.centroid.y
            comx2, comy2 = barc2.centroid.x, barc2.centroid.y
            comxl1, comyl1 = left1.centroid.x,  left1.centroid.y
            comxr1, comyr1 = right1.centroid.x, right1.centroid.y
            comxl2, comyl2 = left2.centroid.x,  left2.centroid.y
            comxr2, comyr2 = right2.centroid.x, right2.centroid.y
        except Exception:
            return _one_rule_fallback(
                x1, y1, lw1, rw1, x2, y2, lw2, rw2,
                lambda1, lambda2, lambda3, alpha, delta, qshift_ratio)

        # COM total ponderado por volumen
        tot_comx = (comx1 * (l1v + r1v) + comx2 * (l2v + r2v)) / original_vtot
        tot_comy = (comy1 * (l1v + r1v) + comy2 * (l2v + r2v)) / original_vtot

        # Detectar intersecciones flanco a flanco
        l1col = r1col = l2col = r2col = False
        try:
            if left1.intersects(left2):   l1col = True;  l2col = True
            if left1.intersects(right2):  l1col = True;  r2col = True
            if right1.intersects(left2):  r1col = True;  l2col = True
            if right1.intersects(right2): r1col = True;  r2col = True
        except Exception:
            l1col = l2col = True  # fallback: fusionar todo

        # Separar colisionantes y no-colisionantes
        colliders_v, colliders_x, colliders_y = [], [], []
        others_v,    others_w,    others_x, others_y = [], [], [], []

        for col, v, w, cx, cy in [
            (l1col, l1v, lw1, comxl1, comyl1),
            (l2col, l2v, lw2, comxl2, comyl2),
            (r1col, r1v, rw1, comxr1, comyr1),
            (r2col, r2v, rw2, comxr2, comyr2),
        ]:
            if col:
                colliders_v.append(v); colliders_x.append(cx); colliders_y.append(cy)
            else:
                others_v.append(v); others_w.append(w)
                others_x.append(cx); others_y.append(cy)

    else:
        # ── Fallback sin Shapely — fusionar todo ───────────────────────────────
        return _one_rule_fallback(
            x1, y1, lw1, rw1, x2, y2, lw2, rw2,
            lambda1, lambda2, lambda3, alpha, delta, qshift_ratio)

    # ── Fallback: Shapely no detectó ninguna intersección flanco-flanco
    # pero centroid_intersect sí — usar fallback con fusión total
    if len(colliders_v) == 0:
        return _one_rule_fallback(
            x1, y1, lw1, rw1, x2, y2, lw2, rw2,
            lambda1, lambda2, lambda3, alpha, delta, qshift_ratio)

    # ── Fusión de colisionantes ────────────────────────────────────────────────
    cvoltot = sum(colliders_v)
    cwtot   = _equiv_width_body(cvoltot, lambda2, lambda3)

    # Evaluar flancos no-colisionantes con gamma_c (C-02)
    final_v, final_x, final_y = list(colliders_v), list(colliders_x), list(colliders_y)
    ts_v, ts_x, ts_y = [], [], []

    for i, (v, w, cx, cy) in enumerate(zip(others_v, others_w, others_x, others_y)):
        wf = _equiv_width_flank(v, lambda2, lambda3)
        if cwtot <= 0:
            ts_v.append(v); ts_x.append(cx); ts_y.append(cy)
            continue
        ratio    = wf / cwtot
        ratioinv = cwtot / wf if wf > 0 else float('inf')
        gc = gamma_c(wf, alpha, delta, lambda1, lambda2, qshift_ratio)
        if ratioinv < gc and ratio < gc:
            final_v.append(v); final_x.append(cx); final_y.append(cy)
        else:
            ts_v.append(v); ts_x.append(cx); ts_y.append(cy)

    # ── Calcular duna fusionada ────────────────────────────────────────────────
    final_v  = np.array(final_v)
    final_x  = np.array(final_x)
    final_y  = np.array(final_y)
    cvoltot2 = float(final_v.sum())

    if cvoltot2 <= 0:
        return [], [], [], []

    col_comx = float((final_x * final_v).sum() / cvoltot2)
    col_comy = float((final_y * final_v).sum() / cvoltot2)

    # C-03: fracción izq/der ponderada por cvols**(1/3)
    eff_cws  = final_v ** (1.0 / 3.0)
    left_mask = final_x <= col_comx
    left_frac  = float(eff_cws[left_mask].sum() / eff_cws.sum()) if eff_cws.sum() > 0 else 0.5
    right_frac = 1.0 - left_frac

    col_lw = _equiv_width_flank(cvoltot2 * left_frac,  lambda2, lambda3)
    col_rw = _equiv_width_flank(cvoltot2 * right_frac, lambda2, lambda3)
    col_w  = (col_lw + col_rw) / 2.0

    # C-05: posición corregida con término geométrico del paper
    col_y = col_comy + lambda1 * col_w / 2.0 - lambda1 * col_w / 8.0

    # ── Dunas "demasiado pequeñas" (ts) — forman dunas separadas ──────────────
    out_xs  = []
    out_ys  = []
    out_lws = []
    out_rws = []

    for i, (v, cx, cy) in enumerate(zip(ts_v, ts_x, ts_y)):
        tsw = _equiv_width_body(v, lambda2, lambda3) / 2.0
        if tsw > 0:
            thew = max(col_lw, col_rw)
            ty   = col_y - lambda2 * thew + np.random.normal(0, 1e-5)
            out_xs.append(cx)
            out_ys.append(ty)
            out_lws.append(tsw)
            out_rws.append(tsw)

    # Agregar duna fusionada principal
    out_xs.append(col_comx)
    out_ys.append(col_y)
    out_lws.append(col_lw)
    out_rws.append(col_rw)

    # C-04: conservar COM total
    all_v  = np.array(list(ts_v) + [cvoltot2])
    all_xs = np.array(out_xs)
    all_ys = np.array(out_ys)

    # Calcular COM actual de los productos
    try:
        from .flux_physics import _barchan_polys as bp
        bscomx = []
        bscomy = []
        for bx, by, blw, brw in zip(out_xs, out_ys, out_lws, out_rws):
            bl, br = bp(bx, by, blw, brw, lambda1, lambda2, alpha, delta)
            if bl is not None and br is not None:
                bu = unary_union([bl, br])
                bscomx.append(bu.centroid.x)
                bscomy.append(bu.centroid.y)
            else:
                bscomx.append(bx)
                bscomy.append(by)
        bscomx = np.array(bscomx)
        bscomy = np.array(bscomy)
        temp_comx = float((all_v / original_vtot * bscomx).sum())
        temp_comy = float((all_v / original_vtot * bscomy).sum())
        diff_x = tot_comx - temp_comx
        diff_y = tot_comy - temp_comy
        out_xs  = [x + diff_x for x in out_xs]
        out_ys  = [y + diff_y for y in out_ys]
    except Exception:
        pass  # si falla el ajuste de COM, usar posiciones sin ajustar

    return out_xs, out_ys, out_lws, out_rws


def _one_rule_fallback(
    x1, y1, lw1, rw1,
    x2, y2, lw2, rw2,
    lambda1, lambda2, lambda3,
    alpha, delta, qshift_ratio,
):
    """Fallback sin Shapely — fusión total conservando volumen y COM."""
    l1v = _vs_total(lw1, rw1, lambda1, lambda2, lambda3)
    r1v = _vs_total(rw1, lw1, lambda1, lambda2, lambda3)
    l2v = _vs_total(lw2, rw2, lambda1, lambda2, lambda3)
    r2v = _vs_total(rw2, lw2, lambda1, lambda2, lambda3)
    vtot = l1v + r1v + l2v + r2v

    com_x = (x1 * (l1v + r1v) + x2 * (l2v + r2v)) / vtot
    com_y = (y1 * (l1v + r1v) + y2 * (l2v + r2v)) / vtot

    contributions = [
        (x1 - lw1/2, l1v), (x1 + rw1/2, r1v),
        (x2 - lw2/2, l2v), (x2 + rw2/2, r2v),
    ]
    v_left  = sum(v for cx, v in contributions if cx <= com_x)
    v_right = vtot - v_left

    new_lw = width_from_volume(v_left,  lambda2, lambda3)
    new_rw = width_from_volume(v_right, lambda2, lambda3)

    w_min_f  = min(new_lw, new_rw)
    w_max_f  = max(new_lw, new_rw)
    gc = gamma_c(w_min_f, alpha, delta, lambda1, lambda2, qshift_ratio)

    if w_min_f <= 0 or (w_max_f / (w_min_f + 1e-30)) <= gc:
        return [com_x], [com_y], [new_lw], [new_rw]
    else:
        w_main = (new_lw + new_rw) / 2.0
        w_sec  = abs(new_lw - new_rw) / 2.0
        xs, ys, lws, rws = [com_x], [com_y], [w_main], [w_main]
        if w_sec > 0:
            offset = lambda2 * w_main
            xs.append(com_x)
            ys.append(com_y - offset)
            lws.append(w_sec)
            rws.append(w_sec)
        return xs, ys, lws, rws


def classify_collision(n_outputs: int) -> str:
    if n_outputs == 1:
        return 'merging'
    elif n_outputs == 2:
        return 'exchange'
    else:
        return 'fragmentation'