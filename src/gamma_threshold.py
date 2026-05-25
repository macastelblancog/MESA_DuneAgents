"""
gamma_threshold.py
Umbral de asimetría máxima γ_c que dispara el calveo.

Implementa Ecuaciones 4–6 de Robson & Baas (2024):
    Ec. 4  γ_c,shift = 1 + α/(λ₂·q̃ − α) + Δ/(2·(λ₂·q̃ − α)·w_min)
    Ec. 5  γ_c,λ     = 2·λ₂/λ₁ − 1
    Ec. 6  γ_c       = min(γ_c,shift, γ_c,λ)

Historial de correcciones:
    v1/doc1 (Bug B-01): lambda1 ausente — γ_c,shift calculado con λ₁=1 implícito.
    v3/doc5 (C-01 parcial): lambda1 añadido en la firma, pero faltaba γ_c,λ.
    ESTA versión: implementa el min() completo tal como el código v3 de agosto 2024.

Referencias:
    Robson & Baas (2023) GRL — Ec. 4 original (sin γ_c,λ)
    Robson & Baas (2024) ESD — Ecs. 5 y 6 (segundo término y combinación)
"""

from __future__ import annotations


def gamma_c(
    w_min: float,
    alpha: float,
    delta: float,
    lambda1: float,
    lambda2: float,
    qshift_ratio: float,
) -> float:
    """Umbral de asimetría máxima γ_c = min(γ_c,shift, γ_c,λ).

    Parámetros
    ----------
    w_min        : ancho del flanco más pequeño [m]
    alpha        : coeficiente de ancho de cuerno (adim), típicamente 0.05
    delta        : offset de ancho de cuerno [m], típicamente 4.6
    lambda1      : ratio longitud cuerpo / ancho total (adim), típicamente 1.0
    lambda2      : ratio longitud cuerno / ancho flanco (adim), típicamente 1.8
    qshift_ratio : ratio qshift / qsat (adim), e.g. 0.10

    Retorna
    -------
    float : umbral γ_c ≥ 1. Cuando lw/rw > γ_c (o rw/lw > γ_c), se dispara calveo.

    Notas
    -----
    - Si el denominador (λ₂·q̃ − α) ≤ 0, γ_c,shift es indefinido y se retorna γ_c,λ.
    - Si γ_c,shift resulta negativo (w_min muy pequeño o qshift_ratio muy pequeño),
      también se retorna γ_c,λ (comportamiento del v3).
    - Para reproducir el código v1/2023 (con B-01 activo): pasar lambda1=1.0 y
      omitir el segundo término; aquí siempre se calcula el mínimo correcto.
    """
    # ── Segundo umbral: geométrico (Ec. 5) ────────────────────────────────────
    gc_lambda = 2.0 * lambda2 / lambda1 - 1.0

    # ── Primer umbral: por flujo lateral (Ec. 4) ──────────────────────────────
    denom = lambda2 * qshift_ratio - alpha
    if denom <= 0.0 or w_min <= 0.0:
        # qshift demasiado pequeño para estabilizar — solo limitado geométricamente
        return gc_lambda

    gc_shift = 1.0 + alpha / denom + delta / (2.0 * denom * w_min)

    if gc_shift <= 0.0:
        return gc_lambda

    # ── Combinación Ec. 6 ─────────────────────────────────────────────────────
    return min(gc_shift, gc_lambda)


def w_min_theoretical(alpha: float, delta: float) -> float:
    """Ancho mínimo teórico por debajo del cual la duna no puede existir.

    Deriva de H = α·W + Δ/2 > 0 cuando W → 0: H → Δ/2.
    La condición es W_horn > 0, que se cumple para W > (Δ/2)/(1 − α).

    Parámetros
    ----------
    alpha : coeficiente horn (0.05)
    delta : offset horn [m] (4.6)

    Retorna
    -------
    float : w_min [m] ≈ 2.42 m con los valores del paper
    """
    return (delta / 2.0) / (1.0 - alpha)