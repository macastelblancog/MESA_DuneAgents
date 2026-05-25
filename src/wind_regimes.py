"""
wind_regimes.py
Régimen de viento — muestreo y conversiones ángulo ↔ vector.

Convención de coordenadas del modelo:
    • Viento primario en dirección −y: vector canónico [0, −1]
    • Eje x: dirección lateral (lw = flanco +x = estribor)
    • Eje y: dirección downwind (y decreciente = barlovento → sotavento)
    • Inyección en y = simlength (borde norte/barlovento)
    • Eliminación en y ≤ 0 (borde sur/sotavento)

GetVector(270°) = [0, −1]  ← dirección primaria del paper
    Verificación: rtheta = 3π/2, cos(3π/2)=0, sin(3π/2)=−1 → [0,−1] ✓

Referencias:
    WinddirectionStuff.py del repositorio v3 (Robson, 2024)
    Paper 2024 §3.1 (bimodal: θ_b = 22.5, 45, 67.5° desde la moda primaria)
"""

from __future__ import annotations
import math
import numpy as np


# ── Conversiones ángulo ↔ vector ─────────────────────────────────────────────

def get_vector(angle_deg: float) -> tuple[float, float]:
    """Convierte ángulo en grados al vector unitario de dirección del viento.

    Parámetros
    ----------
    angle_deg : ángulo en grados en la convención del modelo.
                270° → [0, −1] (primario del paper, downwind en −y).

    Retorna
    -------
    (wx, wy) : vector unitario, norma = 1.0
    """
    rtheta = math.radians(angle_deg)
    wx = math.cos(rtheta)
    wy = math.sin(rtheta)
    # Normalizar (ya es unitario por construcción, pero por robustez numérica)
    norm = math.hypot(wx, wy)
    return (wx / norm, wy / norm)


def get_angle(wind_vec: tuple[float, float]) -> float:
    """Convierte vector de viento al ángulo en radianes en [0, 2π).

    Compatible con WinddirectionStuff.GetAngle() del código v3.
    Para [0, −1] retorna 3π/2. Para [1, 0] retorna 0.

    Implementación con atan2 para evitar el problema de comparación exacta
    con 0: math.cos(3π/2) = −1.84e−16 (no exactamente 0), lo que haría
    fallar la rama `if wx == 0.0` del código original.

    Verificación:
        get_angle([0, −1]) = atan2(−1, 0) = −π/2 + 2π = 3π/2 ✓
        |sin(3π/2)| = 1 → transferencia lateral máxima con viento primario ✓
    """
    wx, wy = wind_vec
    theta = math.atan2(wy, wx)   # retorna en (−π, π]
    if theta < 0.0:
        theta += 2.0 * math.pi   # normalizar a [0, 2π)
    return theta


# ── Clase WindRegime ──────────────────────────────────────────────────────────

class WindRegime:
    """Muestreador de dirección de viento por régimen.

    Soporta:
        'fixed'          : vector constante (sin aleatoriedad)
        'unimodal'       : Gaussiana centrada en mean_deg, σ = std_deg
        'bimodal'        : dos Gaussianas con peso weight1 para la primaria
        'multidirectional': uniforme en [0°, 360°]

    Parámetros del JSON correspondientes (sección "wind"):
        regime           : str
        mean_deg         : ángulo central primario (270° = [0,−1])
        std_deg          : desviación estándar en grados (0 = fijo)
        secondary_deg    : ángulo de la moda secundaria (solo bimodal)
        secondary_std_deg: σ de la moda secundaria
        secondary_weight : fracción del tiempo en moda secundaria (0–1)
                           paper 2024 usa 0.25 (1/4 del año en secundaria)
    """

    _VALID_REGIMES = {'fixed', 'unimodal', 'bimodal', 'multidirectional'}

    def __init__(
        self,
        regime: str = 'unimodal',
        mean_deg: float = 270.0,
        std_deg: float = 3.0,
        secondary_deg: float | None = None,
        secondary_std_deg: float | None = None,
        secondary_weight: float = 0.25,
        rng: np.random.Generator | None = None,
    ):
        if regime not in self._VALID_REGIMES:
            raise ValueError(f"regime debe ser uno de {self._VALID_REGIMES}, recibido '{regime}'")

        self.regime = regime
        self.mean_deg = mean_deg
        self.std_deg = std_deg
        self.secondary_deg = secondary_deg if secondary_deg is not None else mean_deg + 22.5
        self.secondary_std_deg = secondary_std_deg if secondary_std_deg is not None else std_deg
        self.secondary_weight = float(secondary_weight) if secondary_weight is not None else 0.25  # null en JSON unimodal
        self._rng = rng if rng is not None else np.random.default_rng()

    def sample(self) -> tuple[float, float]:
        """Muestrea un vector de viento unitario (wx, wy) para el paso actual.

        Retorna
        -------
        (wx, wy) : vector unitario con ||(wx, wy)|| = 1.0
        """
        if self.regime == 'fixed':
            return get_vector(self.mean_deg)

        if self.regime == 'unimodal':
            if self.std_deg == 0.0:
                theta = self.mean_deg
            else:
                theta = float(self._rng.normal(self.mean_deg, self.std_deg))
            return get_vector(theta)

        if self.regime == 'bimodal':
            # Paper 2024: 75% moda primaria, 25% moda secundaria
            use_primary = self._rng.uniform() > self.secondary_weight
            if use_primary:
                theta = float(self._rng.normal(self.mean_deg, self.std_deg))
            else:
                theta = float(self._rng.normal(self.secondary_deg, self.secondary_std_deg))
            return get_vector(theta)

        if self.regime == 'multidirectional':
            theta = float(self._rng.uniform(0.0, 360.0))
            return get_vector(theta)

        raise ValueError(f"régimen no manejado: {self.regime}")

    @classmethod
    def from_dict(cls, d: dict, rng: np.random.Generator | None = None) -> "WindRegime":
        """Construye WindRegime desde la sección 'wind' del JSON de parámetros."""
        return cls(
            regime=d.get('regime', 'unimodal'),
            mean_deg=float(d.get('mean_deg', 270.0)),
            std_deg=float(d.get('std_deg', 3.0)),
            secondary_deg=d.get('secondary_deg'),
            secondary_std_deg=d.get('secondary_std_deg'),
            secondary_weight=float(d.get('secondary_weight', 0.25)),
            rng=rng,
        )

    def __repr__(self) -> str:
        return (f"WindRegime(regime='{self.regime}', mean={self.mean_deg}°, "
                f"std={self.std_deg}°)")