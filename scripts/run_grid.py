"""
run_grid.py  -  Grid search de parametros con multiprocessing.

Grid piloto: 162 corridas (3×3×3×3×2×1)
Parametros fijos: paper Robson & Baas (2024) — dominio 9km×10km, rho0=37km⁻²

Uso:
    python scripts/run_grid.py
    python scripts/run_grid.py --n-workers 4 --seed 980330
    python scripts/run_grid.py --dt 0.125 --n-steps 2000   # forzar valores fijos
    python scripts/run_grid.py --resume                     # saltar corridas ya existentes
    python scripts/run_grid.py --dry-run                    # mostrar plan sin ejecutar

Nota sobre dt y n_steps adaptativos
-------------------------------------
dt y n_steps se calculan por corrida usando la condicion CFL:

    dt  = CFL * W_eq / v_mig(W_eq)
        = CFL * (W_eq + w0) / (c * qsat)

    n_steps = min(ceil(factor * simlength / (v_mig * dt)), N_STEPS_MAX)

donde:
    CFL          = 0.1   (desplazamiento <= 10% del ancho por paso)
    factor       = 3     (corridas con inyeccion: 3 cruces del dominio)
                   1     (corridas sin inyeccion: q0ratio=0)
    N_STEPS_MAX  = 2000  (cap para corridas muy lentas)

Para corridas sin inyeccion (q0ratio=0), w_ref = w_min*3 (duna pequeña tipica).

Nota sobre summary.parquet
--------------------------
Los workers NO escriben en summary.parquet para evitar race conditions.
Al terminar el pool, el proceso principal llama rebuild_summary() una sola vez.

Nota sobre memoria RAM
-----------------------
- imap_unordered procesa resultados uno a uno sin acumular.
- del model + del storage + gc.collect() en finally del worker.
- maxtasksperchild reinicia workers periodicamente.
- _Tee activado ANTES del pool para capturar todo el output.
"""

import sys
import argparse
import itertools
import math
import time
import gc
import json
from pathlib import Path
from multiprocessing import Pool, current_process


# ── Logging ───────────────────────────────────────────────────────────────────

class _Tee:
    """Redirige stdout a pantalla Y a archivo de log simultaneamente."""
    def __init__(self, log_path: Path):
        self._log    = open(log_path, "w", encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        sys.stdout   = self

    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)

    def flush(self):
        self._stdout.flush()
        self._log.flush()

    def close(self):
        sys.stdout = self._stdout
        self._log.close()

    def fileno(self):
        return self._stdout.fileno()


sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dune_swarm import DuneSwarm
from run_storage import RunStorage


# ── FIXED_PARAMS — paper Robson & Baas (2024) ────────────────────────────────

FIXED_PARAMS = {
    # ── Geometría (Sherman et al. 2021) ──────────────────────────────────────
    "lambda1":      1.0,
    "lambda2_mean": 1.8,
    "lambda3":      1/3,        # paper 2024 §2.1: V_tot = W³/40
    "alpha":        0.05,
    "delta":        4.6,
    # ── Migración (Elbelrhiti et al. 2008) ───────────────────────────────────
    "c":            45.0,
    "w0":           16.6,
    # ── Dominio (paper 2024 §2.5) ────────────────────────────────────────────
    "simwidth":     9000.0,
    "simlength":    10000.0,
    "fieldwidth":   3000.0,
    "fieldlength":  10000.0,
    # ── Inyección (paper 2024 Ec. 8) ─────────────────────────────────────────
    "inject":       True,
    "inject_mode":  "weq",
    "rho0":         37e-6,      # 37 km⁻²
    # ── Modelo ───────────────────────────────────────────────────────────────
    "n_dunes_init": 0,
    "collisions":   True,
    # outflux_mode es dimension del grid — no va aqui
}


# ── Grid piloto (162 corridas) ────────────────────────────────────────────────

PARAM_GRID = {
    "qsat":         [30.0, 79.0, 120.0],       # bajo, paper, alto
    "q0ratio":      [0.0, 0.10, 0.25],          # sin inj, bajo, paper
    "qshift_ratio": [0.0, 0.10, 0.15],          # paper explora estos
    "lambda2_std":  [0.0, 0.25, 0.50],           # sin het, media, alta
    "outflux_mode": ["Hersen", "Duran"],         # paper 2023 vs 2024
    "wind_regime":  ["unimodal"],
}
# 3×3×3×3×2×1 = 162 corridas


# ── Constantes ────────────────────────────────────────────────────────────────

DATA_DIR                    = "resultados/"
LAMBDA2_STD_MAX             = 0.5
LAMBDA2_STD_MIN             = 0.0
_PROGRESS_INTERVAL_FRACTION = 5

# Parametros CFL para dt adaptativo
_CFL          = 0.1    # d_paso <= CFL * W_eq por paso
_FACTOR_INJ   = 3      # cruces del dominio para corridas con inyeccion
_FACTOR_NOINJ = 1      # cruces para corridas sin inyeccion (q0ratio=0)
_N_STEPS_MAX  = 2000   # cap de pasos


# ── Calculo de dt y n_steps adaptativos ──────────────────────────────────────

def calc_dt_nsteps(params: dict) -> tuple[float, int]:
    """Calcula dt y n_steps por corrida usando condicion CFL.

    CFL garantiza que cada duna se mueva como maximo CFL*W_eq por paso,
    evitando inestabilidades numericas con dunas pequeñas y dt grande.

    Para corridas sin inyeccion (q0ratio=0):
        - No existe W_eq fisico (denominador <= 0)
        - Se usa w_min*3 como referencia (duna pequeña tipica)
        - factor = 1 (un cruce del dominio es suficiente)

    Retorna (dt, n_steps).
    """
    qsat      = float(params["qsat"])
    q0ratio   = float(params["q0ratio"])
    alpha     = float(params.get("alpha",     FIXED_PARAMS["alpha"]))
    delta     = float(params.get("delta",     FIXED_PARAMS["delta"]))
    c         = float(params.get("c",         FIXED_PARAMS["c"]))
    w0        = float(params.get("w0",        FIXED_PARAMS["w0"]))
    simlength = float(params.get("simlength", FIXED_PARAMS["simlength"]))

    q0    = q0ratio * qsat
    denom = q0 - alpha * qsat

    if denom <= 0.0:
        # Sin inyeccion: no existe W_eq fisico
        # Usar dt=0.125 (paper) para evitar n_steps siempre capeado
        dt      = 0.125
        w_min   = (delta / 2.0) / (1.0 - alpha)
        v_mig   = c * qsat / (w_min * 3.0 + w0)
        t_cruce = simlength / v_mig
        n_steps = min(int(math.ceil(_FACTOR_NOINJ * t_cruce / dt)), _N_STEPS_MAX)
    else:
        w_ref   = delta * qsat / denom   # W_eq
        v_mig   = c * qsat / (w_ref + w0)
        dt      = _CFL * w_ref / v_mig
        t_cruce = simlength / v_mig
        n_steps = min(int(math.ceil(_FACTOR_INJ * t_cruce / dt)), _N_STEPS_MAX)

    return dt, n_steps


# ── Utilidades ────────────────────────────────────────────────────────────────

def _safe_lambda2_std(std: float) -> float:
    clamped = max(LAMBDA2_STD_MIN, min(LAMBDA2_STD_MAX, std))
    if clamped != std:
        print(
            f"WARN  lambda2_std={std} fuera de rango - "
            f"ajustado a {clamped:.2f}",
            flush=True,
        )
    return clamped


def describe_params(params: dict) -> str:
    return (
        f"qsat={params['qsat']}, "
        f"q0ratio={params['q0ratio']}, "
        f"qshift_ratio={params['qshift_ratio']}, "
        f"lambda2_std={params['lambda2_std']}, "
        f"outflux={params.get('outflux_mode','?')}, "
        f"wind={params['wind_regime']}"
    )


# ── Worker ────────────────────────────────────────────────────────────────────

def run_one(args: tuple) -> tuple:
    """Ejecuta una corrida. dt y n_steps vienen precalculados en params."""
    task_id, total_tasks, params, n_steps, seed = args

    params = {
        **params,
        "lambda2_std": _safe_lambda2_std(params["lambda2_std"]),
        "seed":        seed,
    }

    storage = RunStorage(DATA_DIR)
    run_id  = storage.new_run(params)

    process_name   = current_process().name
    start_time     = time.time()
    progress_every = max(1, n_steps // _PROGRESS_INTERVAL_FRACTION)

    print(
        f"\n>>  [{task_id:>4}/{total_tasks}] {run_id}"
        f" | seed={seed}"
        f" | {describe_params(params)}"
        f" | dt={params['dt']:.4f} n_steps={n_steps}"
        f"  [{process_name}]",
        flush=True,
    )

    model = None
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
        print(f"OK [{task_id:>4}/{total_tasks}] {run_id} - {elapsed:.1f} s",
              flush=True)
        return (run_id, "completed", elapsed, None)

    except Exception as exc:
        elapsed = time.time() - start_time
        print(
            f"!! [{task_id:>4}/{total_tasks}] {run_id}"
            f" - {type(exc).__name__}: {exc}",
            flush=True,
        )
        return (run_id, "failed", elapsed, str(exc))

    finally:
        if model is not None:
            del model
        del storage
        gc.collect()


# ── Plan de corridas ──────────────────────────────────────────────────────────

def build_combinations() -> list[dict]:
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    return [
        {**FIXED_PARAMS, **dict(zip(keys, combo))}
        for combo in itertools.product(*values)
    ]


def build_tasks(
    combos:           list[dict],
    n_replicas:       int,
    base_seed:        int,
    dt_override:      float | None = None,
    n_steps_override: int   | None = None,
) -> list[tuple]:
    """Construye tareas con dt y n_steps adaptativos por corrida.

    Si dt_override y n_steps_override se especifican via CLI,
    sobreescriben el calculo adaptativo para todas las corridas.
    """
    tasks   = []
    total   = len(combos) * n_replicas
    task_id = 1

    for params in combos:
        if dt_override is not None and n_steps_override is not None:
            dt      = dt_override
            n_steps = n_steps_override
        else:
            dt, n_steps = calc_dt_nsteps(params)

        # dt entra en params para que DuneSwarm lo reciba
        params_with_dt = {**params, "dt": dt}

        for replica in range(n_replicas):
            seed = base_seed + replica
            tasks.append((task_id, total, params_with_dt, n_steps, seed))
            task_id += 1

    return tasks


def _completed_keys(data_dir: str) -> set[tuple]:
    """Lee la DB SQLite para identificar corridas ya completadas."""
    import sqlite3
    db_path = Path(data_dir) / "dunas.db"

    if db_path.exists():
        try:
            con = sqlite3.connect(db_path, timeout=10)
            rows = con.execute(
                "SELECT qsat, q0ratio, qshift_ratio, lambda2_std, wind_regime, seed "
                "FROM runs WHERE n_steps_run IS NOT NULL"
            ).fetchall()
            con.close()
            return {
                (float(r[0]), float(r[1]), float(r[2]),
                 float(r[3]), str(r[4]), int(r[5]))
                for r in rows
            }
        except Exception:
            pass

    # Fallback estructura antigua
    completed = set()
    runs_dir  = Path(data_dir) / "runs"
    if not runs_dir.exists():
        return completed
    for params_file in runs_dir.glob("run_*/params.json"):
        try:
            p = json.loads(params_file.read_text(encoding="utf-8"))
            completed.add((
                float(p["qsat"]), float(p["q0ratio"]),
                float(p["qshift_ratio"]), float(p["lambda2_std"]),
                str(p["wind_regime"]), int(p["seed"]),
            ))
        except Exception:
            pass
    return completed


def _task_key(task: tuple) -> tuple:
    _, _, params, _, seed = task
    return (
        float(params["qsat"]),
        float(params["q0ratio"]),
        float(params["qshift_ratio"]),
        float(params["lambda2_std"]),
        str(params["wind_regime"]),
        int(seed),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grid search ABM dunas barchan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dt", type=float, default=None,
        help="dt fijo para todas las corridas (sobreescribe CFL adaptativo).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=None,
        help="n_steps fijo para todas las corridas (sobreescribe CFL adaptativo).",
    )
    parser.add_argument(
        "--n-workers", type=int, default=4,
        help="Procesos paralelos (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, default=980330,
        help="Semilla base (default: 980330)",
    )
    parser.add_argument(
        "--n-replicas", type=int, default=1,
        help="Replicas por combinacion (default: 1)",
    )
    parser.add_argument(
        "--maxtasks", type=int, default=20,
        help="Tareas por worker antes de reiniciarlo (default: 20).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Saltar corridas que ya existen.",
    )
    parser.add_argument(
        "--log-dir", type=str, default="resultados/",
        help="Directorio del log .txt (default: resultados/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostrar plan sin ejecutar.",
    )
    args = parser.parse_args()

    # Validar que dt y n-steps se especifiquen juntos o ninguno
    if (args.dt is None) != (args.n_steps is None):
        parser.error("--dt y --n-steps deben especificarse juntos o ninguno.")

    log_dir  = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_grid_{ts}.txt"

    # Activar Tee ANTES del pool para capturar todo el output
    tee = _Tee(log_path)
    print(f"\n[log] {log_path}", flush=True)

    combos    = build_combinations()
    all_tasks = build_tasks(
        combos,
        n_replicas       = args.n_replicas,
        base_seed        = args.seed,
        dt_override      = args.dt,
        n_steps_override = args.n_steps,
    )

    if args.resume:
        print("\nEscaneando corridas existentes...", flush=True)
        done    = _completed_keys(DATA_DIR)
        tasks   = [t for t in all_tasks if _task_key(t) not in done]
        skipped = len(all_tasks) - len(tasks)
        print(f"  Ya completadas: {skipped}  |  Pendientes: {len(tasks)}", flush=True)
    else:
        tasks   = all_tasks
        skipped = 0

    total_runs = len(combos) * args.n_replicas
    grid_dims  = " x ".join(f"{len(v)} {k}" for k, v in PARAM_GRID.items())

    # Estadisticas de dt y n_steps
    dts    = [t[2]["dt"] for t in tasks]
    nsteps = [t[3]       for t in tasks]

    print("\n" + "=" * 72)
    print("  GRID SEARCH - ABM DUNAS BARCHAN")
    print("=" * 72)
    print(f"  Dimensiones del grid   : {grid_dims}")
    print(f"  Combinaciones          : {len(combos)}")
    print(f"  Replicas por combo     : {args.n_replicas}  (seed base = {args.seed})")
    print(f"  Corridas totales       : {total_runs}")
    if args.resume:
        print(f"  Saltadas (--resume)    : {skipped}")
        print(f"  A ejecutar             : {len(tasks)}")
    if args.dt is not None:
        print(f"  dt (fijo CLI)          : {args.dt}")
        print(f"  n_steps (fijo CLI)     : {args.n_steps}")
    else:
        print(f"  dt adaptativo CFL={_CFL}  : [{min(dts):.4f}, {max(dts):.4f}] años")
        print(f"  n_steps adaptativo     : [{min(nsteps)}, {max(nsteps)}]  "
              f"(cap={_N_STEPS_MAX})")
    print(f"  Workers en paralelo    : {args.n_workers}")
    print(f"  maxtasksperchild       : {args.maxtasks}")
    print(f"  Directorio de salida   : {DATA_DIR}")
    print(f"  w0                     : {FIXED_PARAMS['w0']}  (Elbelrhiti 2008) OK")
    print(f"  rho0                   : {FIXED_PARAMS['rho0']:.1e}  (37 km⁻²) OK")
    print(f"  lambda2_mean           : {FIXED_PARAMS['lambda2_mean']}  (Sherman 2021) OK")
    print(f"  lambda3                : {FIXED_PARAMS['lambda3']:.4f}  (paper 2024) OK")
    print(f"  Dominio                : {FIXED_PARAMS['simwidth']:.0f}×"
          f"{FIXED_PARAMS['simlength']:.0f} m  (paper 2024) OK")
    print(f"  C-04 inject guard      : q0ratio=0 → sin inyeccion OK")
    print("=" * 72)

    if args.dry_run:
        print(f"\n[DRY RUN] - {len(tasks)} corridas pendientes. Primeras 5:")
        for t in tasks[:5]:
            tid, total, params, steps, seed = t
            print(f"  [{tid:>4}/{total}] seed={seed} | dt={params['dt']:.4f} "
                  f"n_steps={steps} | {describe_params(params)}")
        if len(tasks) > 5:
            print(f"  ... ({len(tasks)-5} mas)")
        tee.close()
        return

    if not tasks:
        print("\nOK Todas las corridas ya existen. Nada que ejecutar.")
        tee.close()
        return

    start_time  = time.time()
    n_completed = 0
    n_failed    = 0
    failed_list = []
    elapsed_sum = 0.0

    with Pool(processes=args.n_workers, maxtasksperchild=args.maxtasks) as pool:
        for run_id, status, elapsed, error in pool.imap_unordered(
            run_one, tasks, chunksize=1
        ):
            elapsed_sum += elapsed
            if status == "completed":
                n_completed += 1
            else:
                n_failed += 1
                failed_list.append((run_id, error))

    total_elapsed = time.time() - start_time

    # Rebuild summary
    print(f"\n[rebuild]  Reconstruyendo summary.parquet ({n_completed} corridas)...",
          flush=True)
    try:
        RunStorage(DATA_DIR).rebuild_summary()
        print("OK  summary.parquet actualizado.", flush=True)
    except Exception as exc:
        print(f"WARN  No se pudo reconstruir: {exc}", flush=True)

    # Resumen final
    avg_time = elapsed_sum / n_completed if n_completed > 0 else 0.0

    print("\n" + "=" * 72)
    print("  RESUMEN FINAL")
    print("=" * 72)
    print(f"  Corridas del grid      : {total_runs}")
    if args.resume:
        print(f"  Saltadas (--resume)    : {skipped}")
    print(f"  Ejecutadas esta sesion : {len(tasks)}")
    print(f"  Completadas            : {n_completed}")
    print(f"  Fallidas               : {n_failed}")
    print(f"  Tiempo total           : {total_elapsed:.1f} s  "
          f"({total_elapsed/60:.1f} min)")
    print(f"  Tiempo medio/corrida   : {avg_time:.1f} s")
    print(f"  Resultados guardados   : {DATA_DIR}")
    print("=" * 72)

    tee.close()

    if failed_list:
        print("\n  WARN  Corridas con error:")
        for rid, err in failed_list:
            print(f"     _ {rid}: {err}")
        print()


if __name__ == "__main__":
    main()