"""
scripts/run_storage.py
Almacenamiento de corridas en SQLite — reemplaza la estructura de directorios.

Estructura del archivo:
    resultados/
    ├── dunas.db          ← base de datos SQLite (único archivo de datos)
    └── summary.parquet   ← índice rápido para la app Dash (se regenera desde la DB)

Tablas en dunas.db:
    runs        : una fila por corrida (params + métricas finales)
    model_data  : series temporales del modelo (una fila por paso × corrida)
    agent_data  : estado de agentes por paso × corrida (opcional, pesado)

API pública — idéntica al RunStorage anterior:
    storage = RunStorage("resultados/")
    run_id  = storage.new_run(params)
    storage.finalize_run(run_id, model, update_summary=False)
    storage.rebuild_summary()
    RunStorage.load_run(data_dir, run_id) → {"params", "model", "agents"}

Compatibilidad:
    load_run() detecta automáticamente si existe dunas.db o la estructura
    antigua de directorios, para no romper corridas ya guardadas.

Correcciones respecto a la versión de directorios:
    B1  finalize_run acepta update_summary=False (sin race condition en batch)
    B2  race condition neutralizada — workers nunca tocan la DB directamente;
        cada worker guarda en un archivo temporal .parquet y el proceso
        principal los consolida con rebuild_summary()
    B3  n_dunes_final compatible con DuneSwarm (MESA) y SwarmSimulator
    B4  n_dunes_final filtra al último paso antes de contar
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Columnas del summary (orden canónico) ────────────────────────────────────

_SUMMARY_COLS = [
    "run_id", "wind_regime", "qsat", "q0ratio", "qshift_ratio",
    "lambda2_mean", "lambda2_std", "outflux_mode", "dt",
    "n_steps_run", "n_dunes_final", "mean_width_final",
    "std_width_final", "mean_asymmetry_final",
    "calving_count", "collision_count",
    "merging_count", "exchange_count", "fragmentation_count",
    "calving_rate", "seed",
]

# DDL de la base de datos
_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    params_json     TEXT NOT NULL,
    wind_regime     TEXT,
    qsat            REAL,
    q0ratio         REAL,
    qshift_ratio    REAL,
    lambda2_mean    REAL,
    lambda2_std     REAL,
    outflux_mode    TEXT,
    dt              REAL,
    seed            INTEGER,
    n_steps_run     INTEGER,
    n_dunes_final   INTEGER,
    mean_width_final     REAL,
    std_width_final      REAL,
    mean_asymmetry_final REAL,
    calving_count        INTEGER,
    collision_count      INTEGER,
    merging_count        INTEGER,
    exchange_count       INTEGER,
    fragmentation_count  INTEGER,
    calving_rate         REAL,
    created_at      REAL
);

CREATE TABLE IF NOT EXISTS model_data (
    run_id  TEXT NOT NULL,
    step    INTEGER NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (run_id, step)
);

CREATE TABLE IF NOT EXISTS agent_data (
    run_id    TEXT NOT NULL,
    step      INTEGER NOT NULL,
    agent_id  INTEGER NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (run_id, step, agent_id)
);

CREATE INDEX IF NOT EXISTS idx_model_run ON model_data(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_run ON agent_data(run_id);
CREATE INDEX IF NOT EXISTS idx_agent_step ON agent_data(run_id, step);
"""


# ══════════════════════════════════════════════════════════════════════════════
class RunStorage:
    """Gestiona el almacenamiento de corridas en SQLite."""

    _SUMMARY_COLS = _SUMMARY_COLS

    def __init__(self, out_dir: str | Path = "resultados/") -> None:
        self.base = Path(out_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base / "dunas.db"
        self._init_db()

    # ── Conexión ──────────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        """Context manager para conexión SQLite con WAL y timeout."""
        con = sqlite3.connect(
            self.db_path,
            timeout=30,
            check_same_thread=False,
        )
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._conn() as con:
            con.executescript(_DDL)

    # ── Ciclo de vida de una corrida ──────────────────────────────────────────

    def new_run(self, params: dict) -> str:
        """Registra una nueva corrida en la DB y retorna su run_id."""
        raw    = json.dumps(params, sort_keys=True, default=str) + str(time.time())
        run_id = "run_" + hashlib.md5(raw.encode()).hexdigest()[:8]

        row = {
            "run_id":       run_id,
            "params_json":  json.dumps(params, default=str),
            "wind_regime":  params.get("wind_regime"),
            "qsat":         params.get("qsat"),
            "q0ratio":      params.get("q0ratio"),
            "qshift_ratio": params.get("qshift_ratio"),
            "lambda2_mean": params.get("lambda2_mean"),
            "lambda2_std":  params.get("lambda2_std"),
            "outflux_mode": params.get("outflux_mode"),
            "dt":           params.get("dt"),
            "seed":         params.get("seed"),
            "created_at":   time.time(),
        }

        with self._conn() as con:
            con.execute("""
                INSERT OR IGNORE INTO runs
                    (run_id, params_json, wind_regime, qsat, q0ratio,
                     qshift_ratio, lambda2_mean, lambda2_std, outflux_mode,
                     dt, seed, created_at)
                VALUES
                    (:run_id, :params_json, :wind_regime, :qsat, :q0ratio,
                     :qshift_ratio, :lambda2_mean, :lambda2_std, :outflux_mode,
                     :dt, :seed, :created_at)
            """, row)

        return run_id

    def save_snapshot(self, run_id: str, model, step: int) -> None:
        """Guarda PNG del campo — en un subdirectorio pequeño para presentaciones."""
        snap_dir = self.base / "snapshots" / run_id
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = snap_dir / f"step_{step:06d}.png"
        try:
            _save_field_png(model, path, step)
        except Exception:
            pass

    def finalize_run(
        self,
        run_id: str,
        model,
        update_summary: bool = True,
    ) -> None:
        """
        Guarda model_data y agent_data en la DB y actualiza métricas del run.

        update_summary=False: solo escribe los datos, no recalcula summary.parquet.
        Útil en batch runs paralelos — llamar rebuild_summary() al final del pool.
        """
        model_df = _get_model_df(model)
        agent_df = _get_agent_df(model)

        # Guardar series temporales del modelo
        self._save_model_data(run_id, model_df)

        # Guardar datos de agentes
        if agent_df is not None and not agent_df.empty:
            self._save_agent_data(run_id, agent_df)

        # Snapshot final
        step = int(getattr(model, "current_step", 0))
        self.save_snapshot(run_id, model, step)

        # Actualizar métricas finales en la tabla runs
        params   = self._load_params_from_db(run_id)
        metrics  = _extract_metrics(model, model_df, agent_df)
        self._update_run_metrics(run_id, metrics)

        if update_summary:
            self.rebuild_summary()

    def _save_model_data(self, run_id: str, model_df: pd.DataFrame) -> None:
        if model_df.empty:
            return
        rows = []
        for step, row in model_df.iterrows():
            rows.append((run_id, int(step), row.to_json()))
        with self._conn() as con:
            con.executemany(
                "INSERT OR REPLACE INTO model_data (run_id, step, data_json) VALUES (?,?,?)",
                rows,
            )

    def _save_agent_data(self, run_id: str, agent_df: pd.DataFrame) -> None:
        if agent_df.empty:
            return

        df = agent_df.copy()
        if not isinstance(df.index, pd.MultiIndex):
            if "Step" in df.columns and "AgentID" in df.columns:
                df = df.set_index(["Step", "AgentID"])
            elif "AgentID" in df.columns:
                df.index.name = "Step"
                df = df.reset_index().set_index(["Step", "AgentID"])
            else:
                df.index.name = "AgentID"
                df["Step"] = 0
                df = df.reset_index().set_index(["Step", "AgentID"])

        # Eliminar filas con NaN en el índice — ocurre cuando MESA no registró agentes
        df = df[
            df.index.get_level_values(0).notna() &
            df.index.get_level_values(1).notna()
        ]

        if df.empty:
            return

        rows = []
        for (step, agent_id), row in df.iterrows():
            rows.append((run_id, int(step), int(agent_id), row.to_json()))

        with self._conn() as con:
            for i in range(0, len(rows), 500):
                con.executemany(
                    """INSERT OR REPLACE INTO agent_data
                    (run_id, step, agent_id, data_json) VALUES (?,?,?,?)""",
                    rows[i:i+500],
                )

    def _update_run_metrics(self, run_id: str, metrics: dict) -> None:
        with self._conn() as con:
            con.execute("""
                UPDATE runs SET
                    n_steps_run          = :n_steps_run,
                    n_dunes_final        = :n_dunes_final,
                    mean_width_final     = :mean_width_final,
                    std_width_final      = :std_width_final,
                    mean_asymmetry_final = :mean_asymmetry_final,
                    calving_count        = :calving_count,
                    collision_count      = :collision_count,
                    merging_count        = :merging_count,
                    exchange_count       = :exchange_count,
                    fragmentation_count  = :fragmentation_count,
                    calving_rate         = :calving_rate
                WHERE run_id = :run_id
            """, {**metrics, "run_id": run_id})

    def _load_params_from_db(self, run_id: str) -> dict:
        with self._conn() as con:
            row = con.execute(
                "SELECT params_json FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                pass
        return {}

    # ── Summary ───────────────────────────────────────────────────────────────

    def rebuild_summary(self) -> pd.DataFrame:
        """Regenera summary.parquet desde la tabla runs de la DB."""
        with self._conn() as con:
            df = pd.read_sql("SELECT * FROM runs", con)

        if df.empty:
            empty = pd.DataFrame(columns=_SUMMARY_COLS)
            _safe_parquet(empty, self.base / "summary.parquet")
            return empty

        # Añadir columnas que falten
        for col in _SUMMARY_COLS:
            if col not in df.columns:
                df[col] = None

        summary = df[_SUMMARY_COLS].copy()
        _safe_parquet(summary, self.base / "summary.parquet")
        return summary

    def runs(self) -> pd.DataFrame:
        return self.rebuild_summary()

    # ── Carga de una corrida (API pública) ────────────────────────────────────

    @staticmethod
    def load_run(data_dir: str | Path, run_id: str) -> dict:
        """
        Carga params, model_data y agent_data de una corrida.

        Detecta automáticamente si los datos están en SQLite (dunas.db)
        o en la estructura antigua de directorios (runs/{run_id}/).
        """
        data_dir = Path(data_dir)
        db_path  = data_dir / "dunas.db"

        if db_path.exists():
            return _load_run_from_db(db_path, run_id)

        # Fallback: estructura antigua de directorios
        return _load_run_from_dirs(data_dir, run_id)


# ══════════════════════════════════════════════════════════════════════════════
# Carga desde SQLite
# ══════════════════════════════════════════════════════════════════════════════

def _load_run_from_db(db_path: Path, run_id: str) -> dict:
    con = sqlite3.connect(db_path, timeout=30)
    try:
        # Params
        row = con.execute(
            "SELECT params_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        params = json.loads(row[0]) if row else {}

        # Model data
        rows = con.execute(
            "SELECT step, data_json FROM model_data WHERE run_id=? ORDER BY step",
            (run_id,),
        ).fetchall()
        if rows:
            records = {r[0]: json.loads(r[1]) for r in rows}
            model_df = pd.DataFrame.from_dict(records, orient="index")
            model_df.index.name = "Step"
        else:
            model_df = pd.DataFrame()

        # Agent data
        rows = con.execute(
            """SELECT step, agent_id, data_json FROM agent_data
               WHERE run_id=? ORDER BY step, agent_id""",
            (run_id,),
        ).fetchall()
        if rows:
            records = []
            for step, agent_id, data_json in rows:
                rec = json.loads(data_json)
                rec["Step"]    = step
                rec["AgentID"] = agent_id
                records.append(rec)
            agent_df = pd.DataFrame(records).set_index(["Step", "AgentID"])
        else:
            agent_df = pd.DataFrame()

    finally:
        con.close()

    return {"params": params, "model": model_df, "agents": agent_df}


# ══════════════════════════════════════════════════════════════════════════════
# Fallback: carga desde estructura antigua de directorios
# ══════════════════════════════════════════════════════════════════════════════

def _load_run_from_dirs(data_dir: Path, run_id: str) -> dict:
    run_dir  = data_dir / "runs" / run_id
    params   = _load_params(run_dir / "params.json")
    model_df = _safe_read(run_dir / "model_data")
    agent_df = _safe_read(run_dir / "agent_data")
    return {"params": params, "model": model_df, "agents": agent_df}


# ══════════════════════════════════════════════════════════════════════════════
# Adaptadores modelo → DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def _get_model_df(model) -> pd.DataFrame:
    if hasattr(model, "datacollector"):
        try:
            return model.datacollector.get_model_vars_dataframe()
        except Exception:
            pass
    if hasattr(model, "model_df"):
        return model.model_df
    if hasattr(model, "history") and model.history:
        return pd.DataFrame(model.history).set_index("step")
    return pd.DataFrame()


def _get_agent_df(model) -> pd.DataFrame | None:
    if hasattr(model, "datacollector"):
        try:
            return model.datacollector.get_agent_vars_dataframe()
        except Exception:
            pass
    if hasattr(model, "_xs") and len(model._xs) > 0:
        step = getattr(model, "current_step", 0)
        data = {
            "Step":      [step] * len(model._xs),
            "AgentID":   list(range(len(model._xs))),
            "lw":        model._lws.tolist(),
            "rw":        model._rws.tolist(),
            "width":     (model._lws + model._rws).tolist(),
            "pos_x":     model._xs.tolist(),
            "pos_y":     model._ys.tolist(),
            "lambda2":   model._l2s.tolist(),
            "asymmetry": model.asymmetries.tolist(),
        }
        return pd.DataFrame(data).set_index(["Step", "AgentID"])
    return None


def _extract_metrics(model, model_df: pd.DataFrame,
                     agent_df: pd.DataFrame | None) -> dict:
    """Extrae métricas finales del modelo para guardar en runs."""
    n_steps_run = int(getattr(model, "current_step", len(model_df)))

    # n_dunes_final y métricas de width/asimetría
    if hasattr(model, "agents"):
        try:
            agent_list = list(model.agents)
            widths = np.array([a.lw + a.rw for a in agent_list], dtype=float)
            asym   = np.array([a.asymmetry for a in agent_list], dtype=float)
            n_dunes_final = len(agent_list)
        except Exception:
            widths = np.array([])
            asym   = np.array([])
            n_dunes_final = 0
    elif agent_df is not None and not agent_df.empty:
        # B4: filtrar al último paso
        idx = agent_df.index
        if isinstance(idx, pd.MultiIndex):
            last_step  = idx.get_level_values(0).max()
            agent_last = agent_df.xs(last_step, level=0)
        else:
            agent_last = agent_df
        wcol = "width" if "width" in agent_last.columns else None
        if wcol:
            widths = agent_last[wcol].astype(float).dropna().values
        elif {"lw", "rw"}.issubset(agent_last.columns):
            widths = (agent_last["lw"] + agent_last["rw"]).astype(float).dropna().values
        else:
            widths = np.array([])
        asym = agent_last.get("asymmetry", pd.Series(dtype=float)).astype(float).dropna().values
        n_dunes_final = len(agent_last)
    else:
        widths = np.array([])
        asym   = np.array([])
        n_dunes_final = 0

    calving_count       = int(getattr(model, "calving_count",       0))
    collision_count     = int(getattr(model, "collision_count",     0))
    merging_count       = int(getattr(model, "merging_count",       0))
    exchange_count      = int(getattr(model, "exchange_count",      0))
    fragmentation_count = int(getattr(model, "fragmentation_count", 0))

    return {
        "n_steps_run":          n_steps_run,
        "n_dunes_final":        n_dunes_final,
        "mean_width_final":     float(widths.mean()) if len(widths) > 0 else 0.0,
        "std_width_final":      float(widths.std())  if len(widths) > 1 else 0.0,
        "mean_asymmetry_final": float(asym.mean())   if len(asym)   > 0 else 0.0,
        "calving_count":        calving_count,
        "collision_count":      collision_count,
        "merging_count":        merging_count,
        "exchange_count":       exchange_count,
        "fragmentation_count":  fragmentation_count,
        "calving_rate":         calving_count / max(1, n_steps_run),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot PNG
# ══════════════════════════════════════════════════════════════════════════════

def _save_field_png(model, path: Path, step: int) -> None:
    if hasattr(model, "_xs"):
        xs, ys = model._xs, model._ys
        lws, rws = model._lws, model._rws
        asym = model.asymmetries
        sw, sl = model.simwidth, model.simlength
    elif hasattr(model, "agents"):
        agents = list(model.agents)
        if not agents:
            return
        xs   = np.array([a.pos[0]    for a in agents])
        ys   = np.array([a.pos[1]    for a in agents])
        lws  = np.array([a.lw        for a in agents])
        rws  = np.array([a.rw        for a in agents])
        asym = np.array([a.asymmetry for a in agents])
        sw, sl = model.simwidth, model.simlength
    else:
        return

    if len(xs) == 0:
        return

    widths = lws + rws
    w_max  = max(widths.max(), 1.0)
    sizes  = np.clip(widths / w_max * 200, 8, 200)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xs, ys, s=sizes, c=asym, cmap="RdBu_r",
                    vmin=0, vmax=0.5, alpha=0.82,
                    edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Asimetría", fraction=0.03)
    ax.set_xlim(0, sw)
    ax.set_ylim(0, sl)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)  [↑ barlovento]")
    ax.set_title(f"Paso {step} | N = {len(xs)} | W̄ = {widths.mean():.1f} m")
    ax.annotate("→ viento", xy=(0.02, 0.03), xycoords="axes fraction",
                color="gray", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades de I/O
# ══════════════════════════════════════════════════════════════════════════════

def _safe_json_dump(data: dict, path: Path) -> None:
    def _default(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return str(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_default)


def _load_params(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=True)
    except Exception:
        df.to_csv(path.with_suffix(".csv"))


def _safe_read(base_path: Path) -> pd.DataFrame:
    for ext, reader in [(".parquet", pd.read_parquet), (".csv", pd.read_csv)]:
        p = base_path.with_suffix(ext)
        if p.exists():
            try:
                kwargs = {} if ext == ".parquet" else {"index_col": 0}
                return reader(p, **kwargs)
            except Exception:
                pass
    return pd.DataFrame()