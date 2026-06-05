"""
scripts/generate_demo_data.py
Genera corridas del modelo DuneSwarm y las guarda con RunStorage.

CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR
────────────────────────────────────────
  flux_mode='simple'  →  eliminado. Ahora existe outflux_mode='Hersen'|'Duran'
                          (ver §2.2 Robson & Baas 2024 para la diferencia física).

  wind_regime como string → se expande a parámetros explícitos de WindRegime.
                             Mapeo en WIND_CONFIGS. Las cadenas antiguas se preservan
                             como labels en summary.parquet para compatibilidad con
                             el Dash viewer (usa summary["wind_regime"] para filtros).

  DuneSwarm(**params)  →  DuneSwarm.from_params(full_params_dict, seed=seed).
                           Los parámetros usan los nombres del nuevo contrato
                           (ver src/params_loader.py).

  Grid añade outflux_mode como dimensión científica (no era explorable antes).

  Grids disponibles:
    GRID_PAPER_2024  : reproduce Fig. 3 ESD — barrido de qshift × outflux_mode
    GRID_LAMBDA2     : contribución original — barrido de lambda2_std
    GRID_FULL        : exploración completa (~400 corridas)
    GRID_QUICK       : demo rápida (~16 corridas)

Uso:
    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --grid paper2024 --steps 400
    python scripts/generate_demo_data.py --grid lambda2   --steps 400
    python scripts/generate_demo_data.py --quick
    python scripts/generate_demo_data.py --steps 200 --out resultados/test/
    python scripts/generate_demo_data.py --base-json params/paper_2024_esd.json
"""

import argparse
import copy
import sys
import time
import traceback
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Verificar disponibilidad de mesa ─────────────────────────────────────────
try:
    import mesa
    from mesa.space import ContinuousSpace  # verifica API mesa 3.x
    _MESA_OK = True
except ImportError:
    _MESA_OK = False

from src.params_loader import load_params, params_to_swarm_kwargs
from scripts.run_storage import RunStorage

# ══════════════════════════════════════════════════════════════════════════════
# Expansión de wind_regime string → parámetros explícitos del WindRegime
# ══════════════════════════════════════════════════════════════════════════════

# Mapeo de etiquetas legibles → parámetros de WindRegime.
# Mantiene el campo 'wind_regime' como etiqueta string en los params
# (para que summary.parquet sea filtrable en el Dash viewer) y añade
# los parámetros numéricos que DuneSwarm necesita.
#
# Ángulos en grados en la convención GetVector():
#   270° → [0, −1] (viento primario sur, eje −y del modelo)
#   θ_b separación angular secundaria: se suma a 270°
#   paper 2024 §3.2: bimodal con θ_b = 22.5°, 45°, 67.5° (separación aguda)

WIND_CONFIGS: dict[str, dict] = {
    "unimodal": {
        "wind_regime":           "unimodal",
        "wind_mean_deg":         270.0,
        "wind_std_deg":          3.0,     # paper 2024: σ = 3°, 99.7% dentro de ±9°
        "wind_secondary_deg":    None,
        "wind_secondary_std_deg": None,
        "wind_secondary_weight": None,
    },
    # ── bimodal_acute: θ_b = 22.5° (Fig. 12a paper 2024) ─────────────────
    "bimodal_acute": {
        "wind_regime":            "bimodal",
        "wind_mean_deg":          270.0,
        "wind_std_deg":           3.0,
        "wind_secondary_deg":     292.5,  # 270 + 22.5
        "wind_secondary_std_deg": 3.0,
        "wind_secondary_weight":  0.25,   # 75% primario / 25% secundario
    },
    # ── bimodal_moderate: θ_b = 45° (Fig. 12b paper 2024) ────────────────
    "bimodal_moderate": {
        "wind_regime":            "bimodal",
        "wind_mean_deg":          270.0,
        "wind_std_deg":           3.0,
        "wind_secondary_deg":     315.0,  # 270 + 45
        "wind_secondary_std_deg": 3.0,
        "wind_secondary_weight":  0.25,
    },
    # ── bimodal_obtuse: θ_b = 67.5° (Fig. 12c paper 2024) ────────────────
    "bimodal_obtuse": {
        "wind_regime":            "bimodal",
        "wind_mean_deg":          270.0,
        "wind_std_deg":           3.0,
        "wind_secondary_deg":     337.5,  # 270 + 67.5
        "wind_secondary_std_deg": 3.0,
        "wind_secondary_weight":  0.25,
    },
    # ── multidirectional: uniforme en [0°, 360°] ──────────────────────────
    "multidirectional": {
        "wind_regime":            "multidirectional",
        "wind_mean_deg":          270.0,
        "wind_std_deg":           3.0,
        "wind_secondary_deg":     None,
        "wind_secondary_std_deg": None,
        "wind_secondary_weight":  None,
    },
}


def _expand_wind(regime_str: str) -> dict:
    """Convierte la etiqueta de régimen de viento a parámetros de WindRegime."""
    if regime_str not in WIND_CONFIGS:
        raise ValueError(
            f"Régimen de viento desconocido: '{regime_str}'. "
            f"Disponibles: {list(WIND_CONFIGS)}"
        )
    return copy.deepcopy(WIND_CONFIGS[regime_str])


# ══════════════════════════════════════════════════════════════════════════════
# Parámetros fijos compartidos por todos los grids
# ══════════════════════════════════════════════════════════════════════════════

# Los valores siguen los parámetros físicos validados del paper 2024 (§2).
# El dominio es reducido respecto al paper (9km×10km) para que el grid
# sea computacionalmente manejable.
#
# Para reproducción EXACTA del paper usar:
#   --base-json params/paper_2024_esd.json
# que tiene simwidth=9000, simlength=10000, n_steps=2800.

FIXED_PARAMS: dict = {
    # ── Geometría (paper 2024, tabla §2) ──────────────────────────────────
    "lambda1":                 1.0,
    "lambda2_mean":            1.8,   # Sherman et al. (2021)
    "lambda3":                 1.0 / 3.0,  # paper 2024 → V_tot = W³/40
    "alpha":                   0.05,  # Hersen et al. (2004)
    "delta":                   4.6,   # Hersen et al. (2004)
    # ── Migración (Elbelrhiti et al. 2008) ────────────────────────────────
    "c":                       45.0,
    "w0":                      16.6,
    # ── Tiempo ────────────────────────────────────────────────────────────
    "dt":                      0.125,   # 1/8 año ≈ 45 días (paper 2024 §2.5)
    # ── Dominio reducido para grid search ─────────────────────────────────
    "simwidth":                2000.0,  # paper: 9000
    "simlength":               3000.0,  # paper: 10000
    "fieldwidth":              800.0,   # paper: 3000
    "fieldlength":             3000.0,
    # ── Inyección (paper 2024, Ec. 8) ────────────────────────────────────
    "inject":                  True,
    "inject_mode":             "weq",   # W_eq = Δ·qsat/(q₀ − α·qsat) ≈ 23 m
    "rho0":                    37e-6,   # 37 km⁻² (Elbelrhiti et al. 2008)
    # ── Condiciones iniciales: empezar vacío como en el paper ─────────────
    "n_dunes_init":            0,
    # ── Modelo ────────────────────────────────────────────────────────────
    "collisions":              True,
    "w_transverse_threshold":  60.0,
}

# ══════════════════════════════════════════════════════════════════════════════
# Grids de parámetros
# ══════════════════════════════════════════════════════════════════════════════

# ── Reproducción paper 2024 ESD §3.1.1 (Fig. 3) ──────────────────────────────
# Barrido de qshift × outflux_mode con ρ₀ y qsat fijos del paper.
GRID_PAPER_2024: dict = {
    "qsat":          [79.0],
    "q0ratio":       [0.25],
    "qshift_ratio":  [0.0, 0.05, 0.10, 0.15],  # Fig. 3 paper 2024
    "outflux_mode":  ["Hersen", "Duran"],       # Figs. 1 y 2 separados
    "lambda2_std":   [0.0],
    "wind_regime":   ["unimodal"],
}

# ── Bimodal (paper 2024 §3.2, Figs. 12/13) ───────────────────────────────────
GRID_BIMODAL: dict = {
    "qsat":          [79.0],
    "q0ratio":       [0.25],
    "qshift_ratio":  [0.10],
    "outflux_mode":  ["Duran"],
    "lambda2_std":   [0.0],
    "wind_regime":   ["unimodal", "bimodal_acute", "bimodal_moderate", "bimodal_obtuse"],
}

# ── Contribución original: heterogeneidad de λ₂ ───────────────────────────────
# Cruza qshift × lambda2_std × outflux_mode.
# Hipótesis: λ₂ heterogéneo modifica la distribución de asimetría de manera
# que no puede explicarse solo con qshift (ver paper 2024 §4, última oración).
GRID_LAMBDA2: dict = {
    "qsat":          [79.0],
    "q0ratio":       [0.25],
    "qshift_ratio":  [0.05, 0.10, 0.15],
    "outflux_mode":  ["Duran"],
    "lambda2_std":   [0.0, 0.3, 0.5, 0.8, 1.0],  # variable original del proyecto
    "wind_regime":   ["unimodal", "bimodal_acute"],
}

# ── Grid completo de exploración ─────────────────────────────────────────────
GRID_FULL: dict = {
    "qsat":          [60.0, 79.0, 100.0, 120.0],
    "q0ratio":       [0.15, 0.20, 0.25, 0.30],
    "qshift_ratio":  [0.05, 0.10, 0.15],
    "outflux_mode":  ["Hersen", "Duran"],
    "lambda2_std":   [0.0, 0.3, 0.6],
    "wind_regime":   ["unimodal", "bimodal_acute", "bimodal_obtuse", "multidirectional"],
}

# ── Grid reducido para demo rápida ────────────────────────────────────────────
GRID_QUICK: dict = {
    "qsat":          [79.0],
    "q0ratio":       [0.25],
    "qshift_ratio":  [0.05, 0.10],
    "outflux_mode":  ["Hersen", "Duran"],
    "lambda2_std":   [0.0, 0.5],
    "wind_regime":   ["unimodal"],
}

GRIDS: dict[str, dict] = {
    "paper2024": GRID_PAPER_2024,
    "bimodal":   GRID_BIMODAL,
    "lambda2":   GRID_LAMBDA2,
    "full":      GRID_FULL,
    "quick":     GRID_QUICK,
}

# Pasos por corrida por grid (ajustados al dominio reducido)
GRID_N_STEPS: dict[str, int] = {
    "paper2024": 600,    # 75 años → ver estabilización inicial
    "bimodal":   600,
    "lambda2":   400,    # 50 años
    "full":      400,
    "quick":     200,    # 25 años
}

# ══════════════════════════════════════════════════════════════════════════════
# Construcción de tareas
# ══════════════════════════════════════════════════════════════════════════════

def _grid_to_flat_params(grid_row: dict, fixed: dict) -> dict:
    """Fusiona una combinación de grid con los parámetros fijos.

    La clave 'wind_regime' (string) se expande a parámetros numéricos
    que DuneSwarm entiende. El string se preserva como etiqueta para
    el summary.

    Retorna un dict PLANO de kwargs para DuneSwarm(**kwargs).
    """
    params = copy.deepcopy(fixed)
    params.update(grid_row)

    # Expandir wind_regime string → parámetros de WindRegime
    wr_str = params.pop("wind_regime", "unimodal")
    wind_p = _expand_wind(wr_str)
    params.update(wind_p)          # añade wind_regime, wind_mean_deg, etc.
    # wind_regime ya está dentro de wind_p con el mismo valor string

    return params


def build_tasks(
    grid: dict,
    fixed: dict,
    n_replicas: int,
    base_json: str | None = None,
) -> list[dict]:
    """Construye la lista de tareas (parámetros + semilla) para el grid.

    Si se pasa base_json, los parámetros fijos vienen del JSON y los
    FIXED_PARAMS solo rellenan lo que no está en el JSON.

    Retorna lista de dicts: {"params": flat_dict, "seed": int}
    """
    if base_json:
        json_params = load_params(base_json)
        base_flat = params_to_swarm_kwargs(json_params)
        # fixed puede anular el JSON si es más específico para el grid
        merged_fixed = {**base_flat, **fixed}
    else:
        merged_fixed = fixed

    keys   = list(grid.keys())
    values = list(grid.values())
    tasks  = []

    for combo in product(*values):
        row = dict(zip(keys, combo))
        flat = _grid_to_flat_params(row, merged_fixed)

        for replica in range(n_replicas):
            raw_seed = str(combo) + str(replica)
            seed = int(abs(hash(raw_seed)) % (2**31))
            flat_copy = copy.deepcopy(flat)
            flat_copy["seed"] = seed
            tasks.append({"params": flat_copy, "seed": seed})

    return tasks

# ══════════════════════════════════════════════════════════════════════════════
# Ejecución de una corrida individual
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(params: dict, seed: int):
    """Construye el modelo apropiado según disponibilidad de mesa.

    Si mesa está instalado (mesa 3.x) → DuneSwarm.
    Si no                                               → SwarmSimulator.
    Ambos tienen la misma interfaz: .step(), .run(), .model_df, etc.
    """
    kw = copy.deepcopy(params)
    kw["seed"] = seed

    if _MESA_OK:
        try:
            from src.dune_swarm import DuneSwarm
            return DuneSwarm(**kw)
        except Exception as e:
            print(f"  ⚠ DuneSwarm falló ({e}) — usando SwarmSimulator")

    from notebooks.sim_engine import SwarmSimulator
    return SwarmSimulator(**kw)


def run_one(
    params: dict,
    seed: int,
    n_steps: int,
    storage: RunStorage,
    snapshot_every: int = 100,
) -> str | None:
    """Ejecuta una corrida completa y la guarda en RunStorage.

    Parámetros
    ----------
    params        : dict plano de kwargs para el modelo
    seed          : semilla aleatoria
    n_steps       : número de pasos a simular
    storage       : instancia de RunStorage
    snapshot_every: guardar PNG cada N pasos

    Retorna run_id si tuvo éxito, None si falló.
    """
    try:
        model  = _build_model(params, seed)

        # Parámetros a guardar en params.json
        save_params = {**params, "seed": seed, "n_steps": n_steps}
        run_id = storage.new_run(save_params)

        for step in range(1, n_steps + 1):
            model.step()
            if step % snapshot_every == 0 or step == n_steps:
                storage.save_snapshot(run_id, model, step)

        storage.finalize_run(run_id, model)
        return run_id

    except Exception:
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Punto de entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera corridas del modelo DuneSwarm para el Dash viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Grids disponibles:
  paper2024  Reproduce Fig. 3 ESD 2024 (qshift × outflux_mode)   [~16 corridas]
  bimodal    Reproduce Figs. 12/13 ESD 2024 (régimen bimodal)    [~4 corridas]
  lambda2    Contribución original (λ₂_std × qshift)              [~30 corridas]
  full       Exploración completa                                  [~400 corridas]
  quick      Demo rápida                                           [~8 corridas]

Ejemplos:
  python scripts/generate_demo_data.py --grid quick
  python scripts/generate_demo_data.py --grid paper2024 --steps 600
  python scripts/generate_demo_data.py --grid lambda2 --out resultados/lambda2/
  python scripts/generate_demo_data.py --base-json params/paper_2024_esd.json --grid lambda2
        """,
    )
    parser.add_argument(
        "--grid", choices=list(GRIDS), default="quick",
        help="Grid de parámetros a explorar (default: quick)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Pasos por corrida (anula el default del grid)",
    )
    parser.add_argument(
        "--replicas", type=int, default=1,
        help="Réplicas por combinación de parámetros (default: 1)",
    )
    parser.add_argument(
        "--out", type=str, default="resultados/",
        help="Directorio de salida (default: resultados/)",
    )
    parser.add_argument(
        "--base-json", type=str, default=None,
        dest="base_json",
        help="JSON de parámetros base (anula FIXED_PARAMS)",
    )
    parser.add_argument(
        "--snapshot-every", type=int, default=100,
        dest="snapshot_every",
        help="Guardar PNG cada N pasos (default: 100)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Alias para --grid quick (compatibilidad con versión anterior)",
    )
    args = parser.parse_args()

    if args.quick:
        args.grid = "quick"

    grid    = GRIDS[args.grid]
    n_steps = args.steps if args.steps is not None else GRID_N_STEPS[args.grid]

    tasks = build_tasks(
        grid=grid,
        fixed=FIXED_PARAMS,
        n_replicas=args.replicas,
        base_json=args.base_json,
    )
    total = len(tasks)

    storage = RunStorage(args.out)

    # ── Header ────────────────────────────────────────────────────────────
    print()
    print("═" * 65)
    print("  ABM Dunas Barchán — Generación de datos")
    print("═" * 65)
    if not _MESA_OK:
        print("  ⚠ mesa no instalado → usando SwarmSimulator (aproximación)")
        print("    Para el modelo exacto: pip install mesa shapely")
    else:
        print("  ✅ mesa disponible → usando DuneSwarm (modelo exacto)")
    print(f"  Grid        : {args.grid}  ({total} corridas × {n_steps} pasos)")
    print(f"  dt          : {FIXED_PARAMS['dt']} yr  "
          f"({FIXED_PARAMS['dt'] * n_steps:.1f} años por corrida)")
    print(f"  Dominio     : {FIXED_PARAMS['simwidth']:.0f}m × "
          f"{FIXED_PARAMS['simlength']:.0f}m  "
          f"(paper 2024: 9000×10000 m)")
    print(f"  Salida      : {Path(args.out).resolve()}")
    if args.base_json:
        print(f"  Base JSON   : {args.base_json}")
    print("═" * 65)
    print()

    # ── Corridas ──────────────────────────────────────────────────────────
    ok = 0
    t0 = time.time()

    for i, task in enumerate(tasks, 1):
        p      = task["params"]
        seed   = task["seed"]
        t_run  = time.time()

        wr    = p.get("wind_regime", "?")
        qs    = p.get("qshift_ratio", "?")
        qsat  = p.get("qsat", "?")
        l2std = p.get("lambda2_std", "?")
        mode  = p.get("outflux_mode", "?")

        run_id = run_one(p, seed, n_steps, storage,
                         snapshot_every=args.snapshot_every)
        elapsed = time.time() - t_run

        if run_id:
            ok += 1
            print(f"  [{i:3d}/{total}] ✅  {run_id}  "
                  f"regime={wr:<20} qshift={qs:<5} qsat={qsat:<6} "
                  f"λ₂σ={l2std:<4} mode={mode:<8} ({elapsed:.1f}s)")
        else:
            print(f"  [{i:3d}/{total}] ❌  FALLÓ  "
                  f"regime={wr} qshift={qs} qsat={qsat}")

    # ── Resumen ───────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print()
    print("═" * 65)
    print(f"  Completado : {ok}/{total} corridas OK en {total_time:.0f}s")
    print(f"  summary    : {Path(args.out).resolve() / 'summary.parquet'}")
    print()
    print("  Para visualizar:")
    print(f"    python visualizacion/track_b/app_trackB.py --data {args.out}")
    print("═" * 65)


if __name__ == "__main__":
    main()
