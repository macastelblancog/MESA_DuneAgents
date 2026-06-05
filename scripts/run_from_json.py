#!/usr/bin/env python3
"""
scripts/run_from_json.py
Corre UNA corrida del modelo DuneSwarm desde un JSON de parámetros.
La salida es directamente compatible con la app de visualización.

Compatibilidad con RunStorage
------------------------------
  La corrida se guarda en:
    {out}/runs/{run_id}/params.json
    {out}/runs/{run_id}/model_data.parquet
    {out}/runs/{run_id}/agent_data.parquet
    {out}/runs/{run_id}/snapshots/step_*.png
  Y se registra en:
    {out}/summary.parquet

  Esto permite abrir la corrida inmediatamente con:
    python visualization/app.py --data {out}

Comportamiento de actualización
---------------------------------
  Sin flags   : siempre crea una corrida nueva (se añade al summary).
  --update    : si ya existe una corrida con la misma semilla en el summary,
                la reemplaza (borra el directorio viejo, crea uno nuevo).
  --force     : crea corrida nueva aunque ya exista (alias de comportamiento default
                pero explícito; útil en scripts).

Parámetros JSON
---------------
  El JSON puede ser nested (como params/paper_2024_esd.json) o plano.
  params_to_swarm_kwargs() aplana la estructura antes de pasarla al modelo.
  Los campos que no están en el JSON se toman de FIXED_PARAMS de generate_demo_data.py.

Uso:
    python scripts/run_from_json.py --params params/paper_2023_grl.json
    python scripts/run_from_json.py --params params/paper_2024_esd.json --seed 42
    python scripts/run_from_json.py --params params/paper_2024_esd.json \\
        --n_steps 200 --out resultados/ --update
    python scripts/run_from_json.py --params params/paper_2024_esd.json \\
        --wind unimodal --qshift 0.10 --lambda2_std 0.5 --n_steps 200
"""

import argparse
import copy
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Imports del proyecto ──────────────────────────────────────────────────────
try:
    import mesa
    _MESA_OK = True
except ImportError:
    _MESA_OK = False

from src.params_loader import load_params, params_to_swarm_kwargs
from scripts.run_storage import RunStorage

# ── Reutilizar WIND_CONFIGS y _build_model de generate_demo_data ─────────────
from scripts.generate_demo_data import (
    WIND_CONFIGS,
    FIXED_PARAMS,
    _expand_wind,
    _build_model,
)

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Corre una corrida de DuneSwarm desde un JSON de parámetros.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Corrida básica con defaults del JSON
  python scripts/run_from_json.py --params params/paper_2024_esd.json

  # Anular parámetros específicos
  python scripts/run_from_json.py --params params/paper_2024_esd.json \\
      --seed 42 --n_steps 200 --wind bimodal_acute --lambda2_std 0.5

  # Reemplazar corrida existente con misma semilla
  python scripts/run_from_json.py --params params/paper_2024_esd.json \\
      --seed 42 --n_steps 400 --update

  # Directorio de salida personalizado
  python scripts/run_from_json.py --params params/paper_2024_esd.json \\
      --out resultados/mi_experimento/
        """,
    )

    # ── Parámetros base ───────────────────────────────────────────────────────
    p.add_argument(
        "--params", required=True,
        help="Ruta al JSON de parámetros (nested o plano)",
    )

    # ── Overrides individuales ────────────────────────────────────────────────
    p.add_argument("--seed",        type=int,   default=None,
                   help="Semilla aleatoria (anula la del JSON)")
    p.add_argument("--n_steps",     type=int,   default=None,
                   help="Número de pasos (anula el del JSON)")
    p.add_argument("--wind",        type=str,   default=None,
                   choices=list(WIND_CONFIGS),
                   help="Régimen de viento (unimodal, bimodal_acute, …)")
    p.add_argument("--qsat",        type=float, default=None,
                   help="q_sat (m²/año)")
    p.add_argument("--q0ratio",     type=float, default=None,
                   help="q₀ / q_sat")
    p.add_argument("--qshift",      type=float, default=None,
                   help="q_shift / q_sat (alias de qshift_ratio)")
    p.add_argument("--lambda2_std", type=float, default=None,
                   help="σ de la distribución de λ₂")
    p.add_argument("--outflux",     type=str,   default=None,
                   choices=["Hersen", "Duran"],
                   help="Modo de flujo de salida")

    # ── Salida ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--out", type=str, default="resultados/",
        help="Directorio raíz de RunStorage (default: resultados/)",
    )
    p.add_argument(
        "--snapshot-every", type=int, default=50, dest="snapshot_every",
        help="Guardar PNG cada N pasos (default: 50)",
    )

    # ── Comportamiento de actualización ──────────────────────────────────────
    p.add_argument(
        "--update", action="store_true",
        help="Si existe una corrida con la misma semilla, reemplazarla",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Crear siempre corrida nueva (comportamiento default, flag explícito)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suprimir output de progreso por paso",
    )

    return p.parse_args()


# ── Helpers de params ─────────────────────────────────────────────────────────

def _load_and_flatten(json_path: str) -> dict:
    """
    Carga el JSON (nested o plano) y retorna un dict plano de kwargs.
    Si el JSON ya es plano, params_to_swarm_kwargs lo pasa tal cual.
    """
    raw = load_params(json_path)

    # params_to_swarm_kwargs acepta nested y plano
    try:
        flat = params_to_swarm_kwargs(raw)
    except Exception:
        # Si falla (JSON ya plano), usar directamente
        flat = copy.deepcopy(raw)

    return flat


def _apply_overrides(flat: dict, args: argparse.Namespace) -> dict:
    """Aplica los overrides de CLI sobre el dict plano."""
    flat = copy.deepcopy(flat)

    if args.seed        is not None: flat["seed"]          = args.seed
    if args.n_steps     is not None: flat["n_steps"]       = args.n_steps
    if args.qsat        is not None: flat["qsat"]          = args.qsat
    if args.q0ratio     is not None: flat["q0ratio"]       = args.q0ratio
    if args.qshift      is not None: flat["qshift_ratio"]  = args.qshift
    if args.lambda2_std is not None: flat["lambda2_std"]   = args.lambda2_std
    if args.outflux     is not None: flat["outflux_mode"]  = args.outflux

    # Wind override: expande el régimen y sobreescribe campos de viento
    if args.wind is not None:
        wind_p = _expand_wind(args.wind)
        flat.update(wind_p)

    return flat


def _fill_defaults(flat: dict) -> dict:
    """Rellena con FIXED_PARAMS los campos que faltan en el JSON."""
    result = copy.deepcopy(FIXED_PARAMS)
    result.update(flat)               # el JSON tiene prioridad sobre FIXED_PARAMS

    # Expandir wind_regime si es un string sin parámetros numéricos
    if "wind_mean_deg" not in result:
        wr_str = result.get("wind_regime", "unimodal")
        wind_p = _expand_wind(wr_str if wr_str in WIND_CONFIGS else "unimodal")
        result.update(wind_p)

    # Semilla por defecto determinista si no se especificó
    if "seed" not in result or result["seed"] is None:
        import hashlib, time as _t
        h = hashlib.md5(str(result).encode()).hexdigest()
        result["seed"] = int(h[:8], 16) % (2**31)

    # n_steps por defecto
    if "n_steps" not in result:
        result["n_steps"] = 200

    return result


# ── Detección y limpieza de corridas existentes ───────────────────────────────

def _find_existing_run(data_dir: Path, seed: int) -> str | None:
    """
    Busca en summary.parquet si ya existe una corrida con esta semilla.
    Retorna el run_id si existe, None si no.
    """
    if not _PANDAS_OK:
        return None

    summary_path = data_dir / "summary.parquet"
    if not summary_path.exists():
        return None

    try:
        import pandas as pd
        summary = pd.read_parquet(summary_path)
        if "seed" not in summary.columns or "run_id" not in summary.columns:
            return None
        match = summary[summary["seed"] == seed]
        if match.empty:
            return None
        return str(match.iloc[0]["run_id"])
    except Exception:
        return None


def _remove_run(data_dir: Path, run_id: str) -> None:
    """
    Elimina el directorio de una corrida y su entrada en summary.parquet.
    """
    # Borrar directorio
    run_dir = data_dir / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
        print(f"  🗑  Directorio eliminado: {run_dir}")

    # Actualizar summary.parquet
    if not _PANDAS_OK:
        return
    summary_path = data_dir / "summary.parquet"
    if not summary_path.exists():
        return
    try:
        import pandas as pd
        summary = pd.read_parquet(summary_path)
        summary = summary[summary["run_id"] != run_id]
        summary.to_parquet(summary_path, index=False)
        print(f"  🗑  Entrada eliminada de summary.parquet (run_id={run_id})")
    except Exception as e:
        print(f"  ⚠  No se pudo actualizar summary.parquet: {e}")


# ── Ejecución principal ───────────────────────────────────────────────────────

def run_simulation(
    params: dict,
    storage: RunStorage,
    snapshot_every: int,
    quiet: bool,
) -> str | None:
    """
    Ejecuta la corrida y guarda con RunStorage.
    Retorna run_id si tuvo éxito, None si falló.
    """
    n_steps = params.get("n_steps", 200)
    seed    = params["seed"]

    try:
        model  = _build_model(params, seed)
        run_id = storage.new_run(params)

        t0           = time.time()
        report_every = max(1, n_steps // 20)   # 20 reportes por corrida

        for step in range(1, n_steps + 1):
            model.step()

            if step % snapshot_every == 0 or step == n_steps:
                storage.save_snapshot(run_id, model, step)

            if not quiet and step % report_every == 0:
                n_ag    = len(list(model.agents))
                elapsed = time.time() - t0
                rate    = step / elapsed if elapsed > 0 else 0
                eta     = (n_steps - step) / rate if rate > 0 else 0
                calv    = getattr(model, "calving_count",    "?")
                coll    = getattr(model, "collision_count",  "?")
                print(
                    f"  paso {step:5d}/{n_steps}  |  "
                    f"dunas: {n_ag:4d}  |  "
                    f"calveos: {calv}  colisiones: {coll}  |  "
                    f"ETA: {eta:.0f}s"
                )

        storage.finalize_run(run_id, model)
        return run_id

    except Exception:
        traceback.print_exc()
        return None


# ── Resumen final ─────────────────────────────────────────────────────────────

def _print_summary(model, params: dict, run_id: str, data_dir: Path,
                   elapsed: float) -> None:
    n_ag   = len(list(model.agents)) if model is not None else "?"
    calv   = getattr(model, "calving_count",         "?")
    coll   = getattr(model, "collision_count",        "?")
    merg   = getattr(model, "merging_count",          "?")
    exch   = getattr(model, "exchange_count",         "?")
    frag   = getattr(model, "fragmentation_count",    "?")
    step   = getattr(model, "current_step",           params.get("n_steps", "?"))
    dt     = params.get("dt", 0.125)

    print()
    print("── Resumen ──────────────────────────────────────────────")
    print(f"  run_id             : {run_id}")
    print(f"  Pasos simulados    : {step}")
    print(f"  Años simulados     : {int(step) * dt:.1f}" if isinstance(step, int) else "")
    print(f"  Tiempo de cómputo  : {elapsed:.1f}s")
    print(f"  Dunas activas      : {n_ag}")
    print(f"  Calveos totales    : {calv}")
    print(f"  Colisiones totales : {coll}")
    if coll not in ("?", 0):
        total_c = max(1, coll)
        print(f"    Merging          : {merg} ({100*merg/total_c:.1f}%)")
        print(f"    Exchange         : {exch} ({100*exch/total_c:.1f}%)")
        print(f"    Fragmentation    : {frag} ({100*frag/total_c:.1f}%)")
    print(f"  Salida             : {data_dir / 'runs' / run_id}")
    print(f"  summary.parquet    : {data_dir / 'summary.parquet'}")
    print("─" * 58)
    print()
    print("  Para visualizar:")
    print(f"    python visualization/app.py --data {data_dir}")
    print("─" * 58)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── 1. Cargar y aplanar params ────────────────────────────────────────────
    flat   = _load_and_flatten(args.params)
    flat   = _apply_overrides(flat, args)
    params = _fill_defaults(flat)

    n_steps = params["n_steps"]
    seed    = params["seed"]

    # ── 2. Header ─────────────────────────────────────────────────────────────
    data_dir = Path(args.out).resolve()
    print()
    print("═" * 65)
    print("  ABM Dunas Barchán — Corrida individual")
    print("═" * 65)
    if not _MESA_OK:
        print("  ⚠  mesa no disponible → usando SwarmSimulator")
    else:
        print(" mesa disponible → DuneSwarm")
    print(f"  JSON base   : {args.params}")
    print(f"  Semilla     : {seed}")
    print(f"  Pasos       : {n_steps}  ({n_steps * params.get('dt', 0.125):.1f} años)")
    print(f"  Régimen     : {params.get('wind_regime', '?')}")
    print(f"  qsat        : {params.get('qsat', '?')}  "
          f"q₀/qsat={params.get('q0ratio', '?')}  "
          f"qshift/qsat={params.get('qshift_ratio', '?')}")
    print(f"  λ₂ σ        : {params.get('lambda2_std', '?')}")
    print(f"  outflux     : {params.get('outflux_mode', '?')}")
    print(f"  Dominio     : {params.get('simwidth', '?')}m × {params.get('simlength', '?')}m")
    print(f"  Salida      : {data_dir}")
    print("═" * 65)
    print()

    # ── 3. Detectar y resolver corrida existente ──────────────────────────────
    storage = RunStorage(str(data_dir))

    if args.update:
        existing = _find_existing_run(data_dir, seed)
        if existing:
            print(f"  ↺  Corrida existente encontrada: {existing}")
            print(f"     (semilla={seed}) → será reemplazada")
            _remove_run(data_dir, existing)
            print()
        else:
            print(f"  ℹ  No existe corrida previa con seed={seed} → se creará nueva")
            print()
    elif args.force:
        print("  ℹ  --force: creando corrida nueva siempre")
        print()
    else:
        existing = _find_existing_run(data_dir, seed)
        if existing:
            print(f"  ⚠  Ya existe una corrida con seed={seed} (run_id={existing})")
            print(f"     Usa --update para reemplazarla o --force para añadir una nueva.")
            print(f"     Continuando con corrida nueva...\n")

    # ── 4. Correr y guardar ───────────────────────────────────────────────────
    t_start = time.time()
    run_id  = run_simulation(
        params        = params,
        storage       = storage,
        snapshot_every= args.snapshot_every,
        quiet         = args.quiet,
    )
    elapsed = time.time() - t_start

    if run_id is None:
        print("\n La corrida falló. Revisa el traceback arriba.")
        sys.exit(1)

    print(f"\n Corrida completada en {elapsed:.1f}s  →  run_id: {run_id}")

    # ── 5. Resumen ────────────────────────────────────────────────────────────
    # Reconstruir referencia al modelo para el resumen (ya fue finalizado)
    # Los datos están en disco; usamos lo que RunStorage guardó.
    try:
        import pandas as pd
        agent_df = pd.read_parquet(data_dir / "runs" / run_id / "agent_data.parquet")
        model_df = pd.read_parquet(data_dir / "runs" / run_id / "model_data.parquet")
        n_ag     = len(agent_df.xs(agent_df.index.get_level_values("Step").max(),
                                    level="Step")) if not agent_df.empty else "?"
        calv     = model_df["calving_count"].iloc[-1] if "calving_count" in model_df.columns else "?"
        coll     = model_df.get("collision_count", pd.Series([0])).iloc[-1] if not model_df.empty else "?"
        step_fin = model_df.index[-1] if not model_df.empty else n_steps
        dt       = params.get("dt", 0.125)

        print()
        print("── Resumen ──────────────────────────────────────────────")
        print(f"  run_id             : {run_id}")
        print(f"  Pasos simulados    : {step_fin}")
        print(f"  Años simulados     : {int(step_fin) * dt:.1f}")
        print(f"  Tiempo de cómputo  : {elapsed:.1f}s")
        print(f"  Dunas activas      : {n_ag}")
        print(f"  Calveos totales    : {calv}")
        print(f"  Colisiones totales : {coll}")
        print(f"  Salida             : {data_dir / 'runs' / run_id}")
        print(f"  summary.parquet    : {data_dir / 'summary.parquet'}")
        print("─" * 58)
        print()
        print("  Para visualizar:")
        print(f"    python visualization/app.py --data {data_dir}")
        print("─" * 58)

    except Exception:
        # Si falla la lectura del resumen, simplemente mostrar lo básico
        print()
        print(f"  Salida: {data_dir / 'runs' / run_id}")
        print(f"  python visualization/app.py --data {data_dir}")


if __name__ == "__main__":
    main()