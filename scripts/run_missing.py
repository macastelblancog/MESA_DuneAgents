"""
run_missing.py — Corre solo las combinaciones del grid que faltan en disco.

Uso:
    python scripts/run_missing.py
    python scripts/run_missing.py --n-workers 4 --dry-run

Lógica:
    1. Reconstruye el summary desde los runs existentes en resultados/runs/.
    2. Genera todas las combinaciones del PARAM_GRID con el mismo seed y n_steps
       que se usaron en la corrida original.
    3. Cruza contra el summary para identificar combinaciones ausentes.
    4. Corre solo las pendientes con maxtasksperchild para no repetir el OOM.

El identificador de unicidad de una corrida es la tupla:
    (qsat, q0ratio, qshift_ratio, lambda2_std, wind_regime, seed)
Todos los parámetros FIXED_PARAMS se asumen idénticos entre corridas del mismo grid.
"""

import sys
import argparse
import itertools
import time
import json
from pathlib import Path
from multiprocessing import Pool, current_process

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dune_swarm import DuneSwarm
from scripts.run_storage import RunStorage

# ── Copiar exactamente de run_grid.py ────────────────────────────────────────

FIXED_PARAMS = {
    "lambda1":      1.0,
    "lambda2_mean": 3.0,
    "lambda3":      1/3,
    "alpha":        0.05,
    "delta":        4.6,
    "c":            45.0,
    "w0":           2.0,
    "dt":           0.001,
    "simwidth":     600.0,
    "simlength":    200.0,
    "fieldwidth":   1200.0,
    "fieldlength":  400.0,
    "inject":       True,
    "inject_mode":  "weq",
    "rho0":         4.0e-3,
    "n_dunes_init": 0,
    "collisions":   True,
    "outflux_mode": "Duran",
}

PARAM_GRID = {
    "qsat":         [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
    "q0ratio":      [0.10, 0.15, 0.2, 0.25, 0.35, 0.50],
    "qshift_ratio": [0.01, 0.05, 0.10, 0.15, 0.35, 0.50],
    "lambda2_std":  [0.0, 0.1, 0.2, 0.5],
    "wind_regime":  ["unimodal", "bimodal", "multidirectional"],
}

# Parámetros de la corrida original
ORIGINAL_SEED    = 980330
ORIGINAL_STEPS   = 2000
DATA_DIR         = "resultados/"

# Clave de identificación: estas columnas deben existir en el summary
KEY_COLS = ["qsat", "q0ratio", "qshift_ratio", "lambda2_std", "wind_regime", "seed"]

LAMBDA2_STD_MAX  = 0.6
_PROGRESS_FRAC   = 5


# ── Utilidades ────────────────────────────────────────────────────────────────

def build_all_combos(seed: int) -> list[dict]:
    """Producto cartesiano completo del grid, mezclado con FIXED_PARAMS."""
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = []
    for combo in itertools.product(*values):
        params = {**FIXED_PARAMS, **dict(zip(keys, combo)), "seed": seed}
        combos.append(params)
    return combos


def combo_key(params: dict) -> tuple:
    """Tupla de identificación única de una corrida."""
    return (
        float(params["qsat"]),
        float(params["q0ratio"]),
        float(params["qshift_ratio"]),
        float(params["lambda2_std"]),
        str(params["wind_regime"]),
        int(params["seed"]),
    )


def load_completed_keys(data_dir: str) -> set[tuple]:
    """
    Lee los params.json de cada run_* en disco para construir el conjunto
    de corridas ya completadas. No depende de summary.parquet (puede estar
    corrupto o ausente si el proceso anterior crasheó).
    """
    completed = set()
    runs_dir  = Path(data_dir) / "runs"
    if not runs_dir.exists():
        return completed

    for params_file in sorted(runs_dir.glob("run_*/params.json")):
        try:
            params = json.loads(params_file.read_text(encoding="utf-8"))
            key = (
                float(params["qsat"]),
                float(params["q0ratio"]),
                float(params["qshift_ratio"]),
                float(params["lambda2_std"]),
                str(params["wind_regime"]),
                int(params["seed"]),
            )
            completed.add(key)
        except Exception as exc:
            print(f"⚠️  No se pudo leer {params_file}: {exc}")

    return completed


def describe(params: dict) -> str:
    return (
        f"qsat={params['qsat']}, "
        f"q0ratio={params['q0ratio']}, "
        f"qshift_ratio={params['qshift_ratio']}, "
        f"lambda2_std={params['lambda2_std']}, "
        f"wind={params['wind_regime']}"
    )


# ── Worker (idéntico al de run_grid.py + sin cambios de firma) ───────────────

def run_one(args: tuple) -> dict:
    task_id, total_tasks, params, n_steps = args

    params = dict(params)
    params["lambda2_std"] = max(0.0, min(LAMBDA2_STD_MAX, params["lambda2_std"]))

    storage = RunStorage(DATA_DIR)
    run_id  = storage.new_run(params)

    process_name  = current_process().name
    start_time    = time.time()
    progress_every = max(1, n_steps // _PROGRESS_FRAC)

    print(
        f"\n▶  [{task_id:>4}/{total_tasks}] {run_id}"
        f" | seed={params['seed']}"
        f" | {describe(params)}"
        f" | {n_steps} pasos  [{process_name}]",
        flush=True,
    )

    try:
        model = DuneSwarm(**params)

        for step in range(1, n_steps + 1):
            model.step()
            if step == 1 or step == n_steps or step % progress_every == 0:
                print(
                    f"   [{run_id}] {step}/{n_steps} "
                    f"({100 * step // n_steps:3d}%)",
                    flush=True,
                )

        storage.finalize_run(run_id, model, update_summary=False)

        elapsed = time.time() - start_time
        print(f"✅ [{task_id:>4}/{total_tasks}] {run_id} — {elapsed:.1f} s",
              flush=True)
        return {"run_id": run_id, "status": "completed",
                "elapsed": elapsed, "error": None}

    except Exception as exc:
        elapsed = time.time() - start_time
        print(
            f"❌ [{task_id:>4}/{total_tasks}] {run_id}"
            f" — {type(exc).__name__}: {exc}",
            flush=True,
        )
        return {"run_id": run_id, "status": "failed",
                "elapsed": elapsed, "error": str(exc)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Corre solo las combinaciones del grid ausentes en disco.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="Procesos paralelos (default: 4)",
    )
    parser.add_argument(
        "--maxtasks", type=int, default=50,
        help="Tareas por worker antes de reiniciarlo (default: 50). "
             "Reduce consumo de RAM acumulada.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra qué corridas faltan sin ejecutar nada.",
    )
    args = parser.parse_args()

    # ── Identificar pendientes ────────────────────────────────────────────────
    print("\nEscaneando runs existentes en disco...", flush=True)
    completed_keys = load_completed_keys(DATA_DIR)
    print(f"  Runs completados en disco : {len(completed_keys)}", flush=True)

    all_combos   = build_all_combos(ORIGINAL_SEED)
    total_grid   = len(all_combos)

    missing = [p for p in all_combos if combo_key(p) not in completed_keys]
    n_missing = len(missing)

    print(f"  Combinaciones del grid    : {total_grid}")
    print(f"  Pendientes                : {n_missing}")

    if n_missing == 0:
        print("\n✅ El grid está completo. No hay corridas pendientes.")
        return

    # Construir lista de tareas numeradas
    tasks = [
        (i + 1, n_missing, params, ORIGINAL_STEPS)
        for i, params in enumerate(missing)
    ]

    print(f"\n  Seed original             : {ORIGINAL_SEED}")
    print(f"  Pasos por corrida         : {ORIGINAL_STEPS}")
    print(f"  Workers                   : {args.n_workers}")
    print(f"  maxtasksperchild          : {args.maxtasks}")

    if args.dry_run:
        print(f"\n[DRY RUN] — {n_missing} corridas pendientes. Primeras 10:")
        for t in tasks[:10]:
            tid, total, params, steps = t
            print(f"  [{tid:>4}/{total}] {describe(params)}")
        if n_missing > 10:
            print(f"  ... ({n_missing - 10} más)")
        return

    # ── Pool con fix de memoria ───────────────────────────────────────────────
    start_time = time.time()

    # maxtasksperchild: reinicia el worker cada N tareas → libera RAM al OS.
    # imap_unordered + chunksize=1: no bufferiza el queue; mínimo footprint.
    with Pool(processes=args.n_workers, maxtasksperchild=args.maxtasks) as pool:
        results = list(pool.imap_unordered(run_one, tasks, chunksize=1))

    total_elapsed = time.time() - start_time
    completed = [r for r in results if r["status"] == "completed"]
    failed    = [r for r in results if r["status"] == "failed"]

    # ── Rebuild summary ───────────────────────────────────────────────────────
    print(f"\n🔄  Reconstruyendo summary.parquet ({len(completed_keys) + len(completed)} corridas)...",
          flush=True)
    try:
        storage = RunStorage(DATA_DIR)
        storage.rebuild_summary()
        print("✅  summary.parquet actualizado.", flush=True)
    except Exception as exc:
        print(
            f"⚠️  No se pudo reconstruir summary.parquet: {exc}\n"
            f"    Corre manualmente: python scripts/rebuild_summary.py",
            flush=True,
        )

    # ── Resumen ───────────────────────────────────────────────────────────────
    avg_time = (
        sum(r["elapsed"] for r in completed) / len(completed)
        if completed else 0.0
    )

    print("\n" + "═" * 65)
    print("  RESUMEN FINAL")
    print("═" * 65)
    print(f"  Ya existían en disco      : {len(completed_keys)}")
    print(f"  Pendientes ejecutadas     : {len(completed)}")
    print(f"  Fallidas                  : {len(failed)}")
    print(f"  Tiempo total              : {total_elapsed:.1f} s  "
          f"({total_elapsed/60:.1f} min)")
    print(f"  Tiempo medio/corrida      : {avg_time:.1f} s")
    print("═" * 65)

    if failed:
        print("\n  ⚠  Corridas con error:")
        for r in failed:
            print(f"     • {r['run_id']}: {r['error']}")
        print()


if __name__ == "__main__":
    main()