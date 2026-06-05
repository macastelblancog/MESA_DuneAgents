"""
sensitivity_analysis.py
=======================
Genera figuras de análisis de sensibilidad del grid ABM de dunas barchán.

Fuentes de datos
----------------
- resultados/summary.parquet  : una fila por corrida (métricas finales)
- resultados/dunas.db         : tablas runs, model_data, agent_data

Salida
------
Todas las figuras se guardan en figures/ como PNG (300 dpi) y HTML interactivo.
El HTML permite zoom/hover para presentaciones; el PNG va al documento/PPT.

Uso
---
    python sensitivity_analysis.py
    python sensitivity_analysis.py --data resultados/ --out figures/
    python sensitivity_analysis.py --no-html   # solo PNG

Requiere
--------
    pip install plotly kaleido pandas numpy sqlite3
"""

from __future__ import annotations
import argparse
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Paleta coherente con los papers ──────────────────────────────────────────
# lambda2_std: 3 valores → 3 tonos distinguibles en BW y color
L2_COLORS   = {0.0: "#1a6faf", 0.25: "#e07b39", 0.50: "#2ca02c"}
L2_LABELS   = {0.0: "λ₂_std = 0 (homogéneo)", 0.25: "λ₂_std = 0.25", 0.50: "λ₂_std = 0.50"}

REGIME_COLORS  = {"unimodal": "#2166ac", "bimodal_acute": "#d6604d"}
OUTFLUX_DASH   = {"Hersen": "solid", "Duran": "dash"}

MORPHO_COLORS  = {
    "barchan":    "#2166ac",
    "transverse": "#4dac26",
    "asymmetric": "#e07b39",
    "pre_calving": "#d6604d",
    "ghost":      "#aaaaaa",
}

FIG_WIDTH  = 900
FIG_HEIGHT = 550
FONT_SIZE  = 13
FONT_FAMILY = "Arial"

PLOTLY_TEMPLATE = "plotly_white"

# ── Utilidades ────────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convierte '#rrggbb' a 'rgba(r,g,b,alpha)' válido para Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def save_fig(fig: go.Figure, name: str, out_dir: Path, html: bool = True):
    """Guarda figura como PNG y opcionalmente HTML."""
    png_path = out_dir / f"{name}.png"
    fig.write_image(str(png_path), width=FIG_WIDTH, height=FIG_HEIGHT,
                    scale=2)  # scale=2 → 300 dpi equivalente
    print(f"  PNG  {png_path}")
    if html:
        html_path = out_dir / f"{name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"  HTML {html_path}")


def apply_style(fig: go.Figure, title: str, xlab: str = "", ylab: str = "") -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, font=dict(size=14, family=FONT_FAMILY), x=0.05),
        xaxis_title=xlab,
        yaxis_title=ylab,
        font=dict(family=FONT_FAMILY, size=FONT_SIZE),
        margin=dict(l=60, r=30, t=60, b=60),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=0.5),
    )
    return fig


def moving_average(s: pd.Series, window: int = 20) -> pd.Series:
    return s.rolling(window, min_periods=1, center=True).mean()


def filter_viable(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """Elimina corridas donde n_dunes_final == 0 (campo vacío al final).

    Estas corridas corresponden a q0ratio=0 (sin inyección de nuevas dunas):
    sin flujo entrante, el campo siempre se vacía independientemente de qsat
    o qshift_ratio. Incluirlas en promedios de colisiones o calveos distorsiona
    los resultados porque collision_count=0 no refleja dinámica ausente sino
    condición de borde degenerada.

    Uso: pasar el DataFrame de summary antes de cualquier cálculo de métricas.
    Las corridas eliminadas son documentadas en el título de cada figura.
    """
    n_total  = len(df)
    df_filt  = df[df["n_dunes_final"] > 0].copy()
    n_dropped = n_total - len(df_filt)
    if n_dropped > 0 and context:
        print(f"  [filtro viable] {context}: eliminadas {n_dropped}/{n_total} "
              f"corridas con n_dunes_final=0 (q0ratio=0, campo vacío)", flush=True)
    return df_filt


# Mapeo de nombres canónicos → posibles nombres alternativos en la DB
# El primer nombre que exista en el DataFrame se usará.
_COL_ALIASES: dict[str, list[str]] = {
    "step":                    ["step", "Step", "timestep", "step_count"],
    "N_dunes":                 ["n_dunes", "N_dunes", "ndunes", "num_dunes"],
    "mean_width":              ["mean_width", "meanwidth"],
    "std_width":               ["std_width", "stdwidth"],
    "mean_asymmetry":          ["mean_asymmetry", "meanasymmetry"],
    "calvings_this_step":      ["calvings_this_step", "calving_this_step",
                                "calvings_step", "calving_step", "calvings"],
    "merging_this_step":       ["merging_this_step", "merging_step", "mergings"],
    "exchange_this_step":      ["exchange_this_step", "exchange_step", "exchanges"],
    "fragmentation_this_step": ["fragmentation_this_step", "fragmentation_step",
                                "fragmentations"],
}


def resolve_col(df: pd.DataFrame, canonical: str) -> str | None:
    """Devuelve el nombre real de la columna en df para el nombre canónico dado.

    Retorna None si ningún alias existe en el DataFrame.
    """
    aliases = _COL_ALIASES.get(canonical, [canonical])
    cols_lower = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    return None


def require_col(df: pd.DataFrame, canonical: str, context: str = "") -> str:
    """Como resolve_col pero lanza ValueError descriptivo si no encuentra la columna."""
    col = resolve_col(df, canonical)
    if col is None:
        available = list(df.columns)
        raise ValueError(
            f"Columna '{canonical}' no encontrada en model_data"
            f"{' (' + context + ')' if context else ''}.\n"
            f"  Columnas disponibles: {available}\n"
            f"  Aliases buscados: {_COL_ALIASES.get(canonical, [canonical])}"
        )
    return col


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_summary(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "summary.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}")
    df = pd.read_parquet(path)
    # Normalizar tipos
    df["lambda2_std"]  = df["lambda2_std"].astype(float)
    df["qshift_ratio"] = df["qshift_ratio"].astype(float)
    df["q0ratio"]      = df["q0ratio"].astype(float)
    df["qsat"]         = df["qsat"].astype(float)
    df["wind_regime"]  = df["wind_regime"].astype(str)
    df["outflux_mode"] = df["outflux_mode"].astype(str)
    return df


def load_model_data(data_dir: Path) -> pd.DataFrame:
    """Carga model_data desde SQLite joineada con parámetros de runs.

    Maneja dos esquemas posibles:
    - Esquema columnar: cada métrica es una columna propia.
    - Esquema JSON:     las métricas están serializadas en la columna 'data_json'.
    """
    db = data_dir / "dunas.db"
    if not db.exists():
        raise FileNotFoundError(f"No se encontró {db}")
    con = sqlite3.connect(db)

    # Inspeccionar columnas reales
    raw_cols = pd.read_sql_query(
        "SELECT * FROM model_data LIMIT 1", con
    ).columns.tolist()
    print(f"  [diagnóstico] columnas reales en model_data: {raw_cols}")

    df = pd.read_sql_query(
        """
        SELECT
            m.*,
            r.lambda2_std,
            r.qshift_ratio,
            r.q0ratio,
            r.qsat,
            r.wind_regime,
            r.outflux_mode,
            r.seed
        FROM model_data m
        JOIN runs r ON m.run_id = r.run_id
        """,
        con,
    )
    con.close()

    # ── Esquema JSON: expandir data_json en columnas ──────────────────────────
    if "data_json" in df.columns:
        import json
        print("  [info] expandiendo columna data_json...", flush=True)
        # Parsear cada fila — data_json puede ser str o bytes
        parsed = df["data_json"].apply(
            lambda v: json.loads(v) if isinstance(v, (str, bytes)) else {}
        )
        expanded = pd.json_normalize(parsed)
        # Eliminar data_json y concatenar columnas expandidas
        df = pd.concat(
            [df.drop(columns=["data_json"]).reset_index(drop=True),
             expanded.reset_index(drop=True)],
            axis=1,
        )
        print(f"  [info] columnas tras expansión: {list(df.columns)}", flush=True)

    # Normalizar nombres: todo a minúsculas para búsqueda robusta
    df.columns = [c.lower() for c in df.columns]

    # Deduplicar columnas — el join puede producir 'step' duplicado
    # si runs también tiene una columna step; conservar la primera aparición
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    df["lambda2_std"]  = df["lambda2_std"].astype(float)
    df["qshift_ratio"] = df["qshift_ratio"].astype(float)
    return df


def load_agent_data(data_dir: Path, last_frac: float = 0.20) -> pd.DataFrame:
    """Carga agent_data del último `last_frac` de pasos por corrida.

    Maneja esquema columnar y esquema data_json igual que load_model_data.
    """
    import json as _json

    db = data_dir / "dunas.db"
    con = sqlite3.connect(db)

    # Detectar esquema
    raw_cols = pd.read_sql_query(
        "SELECT * FROM agent_data LIMIT 1", con
    ).columns.tolist()
    has_json = "data_json" in raw_cols
    print(f"  [diagnóstico] columnas reales en agent_data: {raw_cols}", flush=True)

    max_steps = pd.read_sql_query(
        "SELECT run_id, MAX(step) as max_step FROM agent_data GROUP BY run_id", con
    )
    frames = []
    for _, row in max_steps.iterrows():
        cutoff = int(row["max_step"] * (1 - last_frac))
        q = f"""
            SELECT a.*, r.lambda2_std, r.qshift_ratio, r.q0ratio,
                   r.qsat, r.wind_regime, r.outflux_mode, r.seed
            FROM agent_data a
            JOIN runs r ON a.run_id = r.run_id
            WHERE a.run_id = '{row['run_id']}' AND a.step >= {cutoff}
        """
        frames.append(pd.read_sql_query(q, con))
    con.close()

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    # Expandir data_json si existe
    if has_json and "data_json" in df.columns:
        print("  [info] expandiendo data_json de agent_data...", flush=True)
        parsed   = df["data_json"].apply(
            lambda v: _json.loads(v) if isinstance(v, (str, bytes)) else {}
        )
        expanded = pd.json_normalize(parsed)
        df = pd.concat(
            [df.drop(columns=["data_json"]).reset_index(drop=True),
             expanded.reset_index(drop=True)],
            axis=1,
        )
        print(f"  [info] columnas agent_data tras expansión: {list(df.columns)}", flush=True)

    df.columns = [c.lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    df["lambda2_std"] = df["lambda2_std"].astype(float)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPO A — Summary (una fila por corrida)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_A1_heatmap_calving(summary: pd.DataFrame, out_dir: Path, html: bool):
    """A1: Heatmap tasa de calveo — lambda2_std × qshift_ratio, faceta wind_regime."""
    df = filter_viable(summary, "A1")
    df["calving_rate"] = df["calving_count"] / df["n_steps_run"].clip(lower=1)

    regimes = df["wind_regime"].unique()
    n = len(regimes)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[f"Viento: {r}" for r in regimes],
        shared_yaxes=True,
        horizontal_spacing=0.14,
    )
    for col, regime in enumerate(regimes, 1):
        sub = df[df["wind_regime"] == regime]
        pivot = sub.groupby(["lambda2_std", "qshift_ratio"])["calving_rate"].mean().unstack()
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=[f"{v:.2f}" for v in pivot.columns],
                y=[f"{v:.2f}" for v in pivot.index],
                colorscale="YlOrRd",
                showscale=(col == n),
                colorbar=dict(title="cv/paso", len=0.75, thickness=14),
                text=np.round(pivot.values, 3),
                texttemplate="%{text}",
                textfont=dict(size=10),
            ),
            row=1, col=col,
        )
        fig.update_xaxes(title_text="q<sub>shift</sub>", row=1, col=col)
        fig.update_yaxes(title_text="λ₂ std" if col == 1 else "", row=1, col=col)
    apply_style(fig,
                "A1 — Tasa de calveo: λ₂ std × q<sub>shift</sub>"
                "  (cv/paso = calveos por paso)")
    save_fig(fig, "A1_heatmap_calving_rate", out_dir, html)


def fig_A2_heatmap_cv_width(summary: pd.DataFrame, out_dir: Path, html: bool):
    """A2: Heatmap CV de anchos — lambda2_std × qshift_ratio."""
    df = filter_viable(summary, "A2")
    df["cv_width"] = df["std_width_final"] / df["mean_width_final"].clip(lower=0.01)

    regimes = df["wind_regime"].unique()
    n = len(regimes)
    fig = make_subplots(rows=1, cols=n,
                        subplot_titles=[f"Viento: {r}" for r in regimes],
                        shared_yaxes=True,
                        horizontal_spacing=0.14)
    for col, regime in enumerate(regimes, 1):
        sub = df[df["wind_regime"] == regime]
        pivot = sub.groupby(["lambda2_std", "qshift_ratio"])["cv_width"].mean().unstack()
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=[f"{v:.2f}" for v in pivot.columns],
                y=[f"{v:.2f}" for v in pivot.index],
                colorscale="Blues",
                showscale=(col == n),
                colorbar=dict(title="CV", len=0.75, thickness=14),
                text=np.round(pivot.values, 3),
                texttemplate="%{text}",
                textfont=dict(size=10),
            ),
            row=1, col=col,
        )
        fig.update_xaxes(title_text="q<sub>shift</sub>", row=1, col=col)
        fig.update_yaxes(title_text="λ₂ std" if col == 1 else "", row=1, col=col)
    apply_style(fig,
                "A2 — CV de anchos: λ₂ std × q<sub>shift</sub>"
                "  (CV = std/media del ancho)")
    save_fig(fig, "A2_heatmap_cv_width", out_dir, html)


def fig_A3_heatmap_collisions(summary: pd.DataFrame, out_dir: Path, html: bool):
    """A3: Tres heatmaps de tipos de colisión normalizados por pasos.

    Cada panel tiene su propia colorbar posicionada individualmente para
    evitar solapamiento. Abreviatura 'ev/paso' explicada en título.
    """
    df = filter_viable(summary, "A3")
    for col_name in ["merging_count", "exchange_count", "fragmentation_count"]:
        df[col_name + "_rate"] = df[col_name] / df["n_steps_run"].clip(lower=1)

    metrics = [
        ("merging_count_rate",       "Fus.",   "Greens"),   # Fusión
        ("exchange_count_rate",      "Int.",   "Blues"),    # Intercambio
        ("fragmentation_count_rate", "Frag.",  "Reds"),     # Fragmentación
    ]
    # Posiciones x de cada colorbar (xref="paper"): al extremo derecho de cada panel
    # 3 paneles con horizontal_spacing=0.18 → paneles en ~[0, 0.27], [0.37, 0.63], [0.73, 1.0]
    cb_x_positions = [0.28, 0.64, 1.01]
    cb_labels      = ["Fus. (ev/paso)", "Int. (ev/paso)", "Frag. (ev/paso)"]

    regimes = df["wind_regime"].unique()
    for regime in regimes:
        sub = df[df["wind_regime"] == regime]
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f"{m[1]}  ({regime})" for m in metrics],
            shared_yaxes=True,
            horizontal_spacing=0.18,
        )
        for c, (metric, short_label, cscale) in enumerate(metrics, 1):
            pivot = sub.groupby(["lambda2_std", "qshift_ratio"])[metric].mean().unstack()
            zmax  = pivot.values.max() if pivot.values.max() > 0 else 1
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=[f"{v:.2f}" for v in pivot.columns],
                    y=[f"{v:.2f}" for v in pivot.index],
                    colorscale=cscale,
                    showscale=True,
                    zmin=0, zmax=zmax,
                    colorbar=dict(
                        title=dict(text=cb_labels[c - 1], side="right"),
                        len=0.72,
                        thickness=12,
                        x=cb_x_positions[c - 1],
                        xpad=4,
                    ),
                    text=np.round(pivot.values, 4),
                    texttemplate="%{text}",
                    textfont=dict(size=9),
                ),
                row=1, col=c,
            )
            fig.update_xaxes(title_text="q<sub>shift</sub>", row=1, col=c)
            fig.update_yaxes(title_text="λ₂ std" if c == 1 else "", row=1, col=c)

        apply_style(
            fig,
            f"A3 — Tipos de colisión por paso ({regime})"
            "  [Fus.=fusión · Int.=intercambio · Frag.=fragmentación · ev/paso=eventos/paso]",
        )
        fig.update_layout(margin=dict(r=90))   # espacio para la colorbar más a la derecha
        save_fig(fig, f"A3_heatmap_collision_types_{regime}", out_dir, html)


def fig_A4_scatter_ndunes_calving(summary: pd.DataFrame, out_dir: Path, html: bool):
    """A4: N_dunes_final vs tasa de calveo, coloreado por lambda2_std."""
    df = filter_viable(summary, "A4")
    df["calving_rate"] = df["calving_count"] / df["n_steps_run"].clip(lower=1)
    df["lambda2_std_str"] = df["lambda2_std"].map(L2_LABELS)

    fig = px.scatter(
        df,
        x="calving_rate",
        y="n_dunes_final",
        color="lambda2_std_str",
        symbol="wind_regime",
        facet_col="outflux_mode",
        color_discrete_map={v: L2_COLORS[k] for k, v in L2_LABELS.items()},
        labels={
            "calving_rate": "cv/paso (calveos/paso)",
            "n_dunes_final": "N dunas final",
            "lambda2_std_str": "Het. λ₂",
            "wind_regime": "Régimen",
        },
        opacity=0.75,
        size_max=10,
    )
    apply_style(fig, "A4 — Población final vs tasa de calveo  (cv/paso = calveos por paso)")
    save_fig(fig, "A4_scatter_ndunes_calving", out_dir, html)


def fig_A5_heatmap_collision_fraction(summary: pd.DataFrame, out_dir: Path, html: bool):
    """A5: Fracción de cada tipo de colisión sobre total de colisiones."""
    df = filter_viable(summary, "A5")
    total = df["collision_count"].clip(lower=1)
    df["frac_merging"]       = df["merging_count"]       / total
    df["frac_exchange"]      = df["exchange_count"]       / total
    df["frac_fragmentation"] = df["fragmentation_count"]  / total

    metrics = [
        ("frac_merging",       "Fus. %",   "Greens"),
        ("frac_exchange",      "Int. %",   "Blues"),
        ("frac_fragmentation", "Frag. %",  "Oranges"),
    ]
    cb_x_positions = [0.28, 0.64, 1.01]
    cb_labels      = ["Fus. (%)", "Int. (%)", "Frag. (%)"]

    regimes = df["wind_regime"].unique()
    for regime in regimes:
        sub = df[df["wind_regime"] == regime]
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f"{m[1]}  ({regime})" for m in metrics],
            shared_yaxes=True,
            horizontal_spacing=0.18,
        )
        for c, (metric, short_label, cscale) in enumerate(metrics, 1):
            pivot = sub.groupby(["lambda2_std", "qshift_ratio"])[metric].mean().unstack()
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values * 100,
                    x=[f"{v:.2f}" for v in pivot.columns],
                    y=[f"{v:.2f}" for v in pivot.index],
                    colorscale=cscale,
                    showscale=True,
                    zmin=0, zmax=100,
                    colorbar=dict(
                        title=dict(text=cb_labels[c - 1], side="right"),
                        len=0.72,
                        thickness=12,
                        x=cb_x_positions[c - 1],
                        xpad=4,
                    ),
                    text=np.round(pivot.values * 100, 1),
                    texttemplate="%{text}%",
                    textfont=dict(size=9),
                ),
                row=1, col=c,
            )
            fig.update_xaxes(title_text="q<sub>shift</sub>", row=1, col=c)
            fig.update_yaxes(title_text="λ₂ std" if c == 1 else "", row=1, col=c)

        apply_style(
            fig,
            f"A5 — Fracción de tipos de colisión ({regime})"
            "  [Fus.=fusión · Int.=intercambio · Frag.=fragmentación]",
        )
        fig.update_layout(margin=dict(r=90))
        save_fig(fig, f"A5_collision_fraction_{regime}", out_dir, html)


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPO B — model_data (series temporales)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_B1_ma_calvings(model_data: pd.DataFrame, out_dir: Path, html: bool,
                        window: int = 20):
    """B1: Promedio móvil de calveos por paso — una traza por lambda2_std."""
    df = model_data.copy()
    step_col    = require_col(df, "step",               "B1")
    calving_col = require_col(df, "calvings_this_step", "B1")

    max_step = df.groupby("run_id")[step_col].transform("max")
    df["step_norm"] = df[step_col] / max_step.clip(lower=1)
    df["step_bin"]  = pd.cut(df["step_norm"], bins=50, labels=False)

    regimes = df["wind_regime"].unique()
    for regime in regimes:
        sub = df[df["wind_regime"] == regime]
        fig = go.Figure()
        for l2_std in sorted(sub["lambda2_std"].unique()):
            grp = sub[sub["lambda2_std"] == l2_std]
            ts = grp.groupby("step_bin")[calving_col].mean().reset_index()
            ts["ma"] = moving_average(ts[calving_col], window=5)
            fig.add_trace(go.Scatter(
                x=ts["step_bin"] / 50,
                y=ts["ma"],
                name=L2_LABELS.get(l2_std, f"λ₂_std={l2_std}"),
                line=dict(color=L2_COLORS.get(l2_std, "#888"), width=2),
                mode="lines",
            ))
        apply_style(fig,
                    f"B1 — Promedio móvil de calveos por paso ({regime})",
                    "Fracción del tiempo simulado", "Calveos por paso (MA)")
        save_fig(fig, f"B1_ma_calvings_{regime}", out_dir, html)


def fig_B2_ndunes_evolution(model_data: pd.DataFrame, out_dir: Path, html: bool):
    """B2: Evolución de N_dunes — media ± std entre réplicas por lambda2_std."""
    df = model_data.copy()
    step_col   = require_col(df, "step",    "B2")
    ndunes_col = require_col(df, "N_dunes", "B2")

    max_step = df.groupby("run_id")[step_col].transform("max")
    df["step_norm"] = df[step_col] / max_step.clip(lower=1)
    df["step_bin"]  = pd.cut(df["step_norm"], bins=50, labels=False)

    regimes = df["wind_regime"].unique()
    for regime in regimes:
        sub = df[df["wind_regime"] == regime]
        fig = go.Figure()
        for l2_std in sorted(sub["lambda2_std"].unique()):
            grp = sub[sub["lambda2_std"] == l2_std]
            ts = grp.groupby("step_bin")[ndunes_col].agg(["mean", "std"]).reset_index()
            color = L2_COLORS.get(l2_std, "#888888")
            label = L2_LABELS.get(l2_std, f"λ₂_std={l2_std}")
            x = ts["step_bin"] / 50
            fill_color = hex_to_rgba(color, 0.15)
            fig.add_trace(go.Scatter(
                x=pd.concat([x, x[::-1]]),
                y=pd.concat([ts["mean"] + ts["std"],
                             (ts["mean"] - ts["std"])[::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=x, y=ts["mean"],
                name=label,
                line=dict(color=color, width=2),
                mode="lines",
            ))
        apply_style(fig,
                    f"B2 — Evolución de N dunas ({regime})",
                    "Fracción del tiempo simulado", "N dunas (media ± std)")
        save_fig(fig, f"B2_ndunes_evolution_{regime}", out_dir, html)


def fig_B3_width_evolution(model_data: pd.DataFrame, out_dir: Path, html: bool):
    """B3: Evolución de mean_width y std_width — subplots por lambda2_std."""
    df = model_data.copy()
    step_col       = require_col(df, "step",       "B3")
    mean_width_col = require_col(df, "mean_width", "B3")
    std_width_col  = require_col(df, "std_width",  "B3")

    max_step = df.groupby("run_id")[step_col].transform("max")
    df["step_norm"] = df[step_col] / max_step.clip(lower=1)
    df["step_bin"]  = pd.cut(df["step_norm"], bins=50, labels=False)

    regimes = df["wind_regime"].unique()
    for regime in regimes:
        sub = df[df["wind_regime"] == regime]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Ancho medio (m)",
                                            "Desv. estándar de anchos (m)"])
        for l2_std in sorted(sub["lambda2_std"].unique()):
            grp = sub[sub["lambda2_std"] == l2_std]
            ts_mean = grp.groupby("step_bin")[mean_width_col].mean().reset_index()
            ts_std  = grp.groupby("step_bin")[std_width_col].mean().reset_index()
            color = L2_COLORS.get(l2_std, "#888")
            label = L2_LABELS.get(l2_std, f"λ₂_std={l2_std}")
            x = ts_mean["step_bin"] / 50
            fig.add_trace(go.Scatter(x=x, y=ts_mean[mean_width_col],
                                     name=label, line=dict(color=color, width=2),
                                     legendgroup=label),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=ts_std[std_width_col],
                                     name=label,
                                     line=dict(color=color, width=2, dash="dash"),
                                     legendgroup=label, showlegend=False),
                          row=2, col=1)
        fig.update_xaxes(title_text="Fracción del tiempo simulado", row=2)
        apply_style(fig, f"B3 — Evolución de anchos ({regime})")
        save_fig(fig, f"B3_width_evolution_{regime}", out_dir, html)


def fig_B4_collision_breakdown(model_data: pd.DataFrame, out_dir: Path, html: bool):
    """B4: Desglose de tipos de evento por paso (MA) — una figura por wind_regime × lambda2_std."""
    df = model_data.copy()
    step_col = require_col(df, "step", "B4")

    # Resolver nombres reales de las columnas de eventos
    event_defs = [
        ("calvings_this_step",      "Calveo",        "#d6604d"),
        ("merging_this_step",       "Fusión",         "#2ca02c"),
        ("exchange_this_step",      "Intercambio",    "#1a6faf"),
        ("fragmentation_this_step", "Fragmentación",  "#e07b39"),
    ]
    event_cols = []
    for canonical, label, color in event_defs:
        real = resolve_col(df, canonical)
        if real is not None:
            event_cols.append((real, label, color))
        else:
            print(f"  WARN B4: columna '{canonical}' no encontrada, se omite")

    if not event_cols:
        print("  WARN B4: ninguna columna de eventos encontrada, se omite la figura")
        return

    max_step = df.groupby("run_id")[step_col].transform("max")
    df["step_norm"] = df[step_col] / max_step.clip(lower=1)
    df["step_bin"]  = pd.cut(df["step_norm"], bins=50, labels=False)

    regimes = df["wind_regime"].unique()
    for regime in regimes:
        sub = df[df["wind_regime"] == regime]
        for l2_std in sorted(sub["lambda2_std"].unique()):
            grp = sub[sub["lambda2_std"] == l2_std]
            fig = go.Figure()
            for col, label, color in event_cols:
                ts = grp.groupby("step_bin")[col].mean().reset_index()
                ts["ma"] = moving_average(ts[col], window=5)
                fig.add_trace(go.Scatter(
                    x=ts["step_bin"] / 50, y=ts["ma"],
                    name=label, line=dict(color=color, width=2),
                    mode="lines",
                ))
            l2_label = L2_LABELS.get(l2_std, f"λ₂_std={l2_std}")
            apply_style(fig,
                        f"B4 — Eventos por paso: {regime} | {l2_label}",
                        "Fracción del tiempo simulado", "Eventos por paso (MA)")
            save_fig(fig, f"B4_events_{regime}_l2std{l2_std:.2f}", out_dir, html)


def fig_B5_collision_fractions_bar(summary: pd.DataFrame, out_dir: Path, html: bool):
    """B5: Fracción de cada tipo de colisión vs lambda2_std — barras agrupadas."""
    df = filter_viable(summary, "B5")
    total = df["collision_count"].clip(lower=1)
    df["Fusión"]         = df["merging_count"]       / total * 100
    df["Intercambio"]    = df["exchange_count"]       / total * 100
    df["Fragmentación"]  = df["fragmentation_count"]  / total * 100

    agg = (df.groupby(["wind_regime", "lambda2_std"])
             [["Fusión", "Intercambio", "Fragmentación"]]
             .mean()
             .reset_index())
    melted = agg.melt(id_vars=["wind_regime", "lambda2_std"],
                      var_name="tipo", value_name="fraccion_pct")
    melted["lambda2_std_str"] = melted["lambda2_std"].map(L2_LABELS)

    fig = px.bar(
        melted,
        x="lambda2_std_str",
        y="fraccion_pct",
        color="tipo",
        barmode="group",
        facet_col="wind_regime",
        color_discrete_map={
            "Fusión": "#2ca02c",
            "Intercambio": "#1a6faf",
            "Fragmentación": "#e07b39",
        },
        labels={
            "lambda2_std_str": "Heterogeneidad λ₂",
            "fraccion_pct": "% del total de colisiones",
            "tipo": "Tipo",
        },
    )
    apply_style(fig, "B5 — Fracción de tipos de colisión vs heterogeneidad λ₂")
    save_fig(fig, "B5_collision_fractions_bar", out_dir, html)


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPO C — agent_data (estado final)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_C1_width_distribution(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """C1: Distribución de anchos al final — histograma superpuesto por lambda2_std."""
    regimes = agent_final["wind_regime"].unique()
    for regime in regimes:
        sub = agent_final[agent_final["wind_regime"] == regime]
        fig = go.Figure()
        for l2_std in sorted(sub["lambda2_std"].unique()):
            grp = sub[sub["lambda2_std"] == l2_std]
            fig.add_trace(go.Histogram(
                x=grp["width"],
                name=L2_LABELS.get(l2_std, f"λ₂_std={l2_std}"),
                histnorm="probability density",
                opacity=0.6,
                marker_color=L2_COLORS.get(l2_std, "#888"),
                nbinsx=40,
            ))
        fig.update_layout(barmode="overlay")
        apply_style(fig,
                    f"C1 — Distribución de anchos al final ({regime})",
                    "Ancho total (m)", "Densidad de probabilidad")
        save_fig(fig, f"C1_width_distribution_{regime}", out_dir, html)


def fig_C2_asymmetry_distribution(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """C2: Distribución de asymmetry al final."""
    regimes = agent_final["wind_regime"].unique()
    for regime in regimes:
        sub = agent_final[agent_final["wind_regime"] == regime]
        fig = go.Figure()
        for l2_std in sorted(sub["lambda2_std"].unique()):
            grp = sub[sub["lambda2_std"] == l2_std]
            fig.add_trace(go.Histogram(
                x=grp["asymmetry"],
                name=L2_LABELS.get(l2_std, f"λ₂_std={l2_std}"),
                histnorm="probability density",
                opacity=0.6,
                marker_color=L2_COLORS.get(l2_std, "#888"),
                nbinsx=35,
            ))
        fig.update_layout(barmode="overlay")
        apply_style(fig,
                    f"C2 — Distribución de asimetría al final ({regime})",
                    "Índice de asimetría", "Densidad de probabilidad")
        save_fig(fig, f"C2_asymmetry_distribution_{regime}", out_dir, html)


def fig_C3_morphotype_composition(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """C3: Composición morfológica — barras apiladas normalizadas."""
    grp = (agent_final.groupby(["wind_regime", "lambda2_std", "morphotype"])
                      .size()
                      .reset_index(name="count"))
    totals = grp.groupby(["wind_regime", "lambda2_std"])["count"].transform("sum")
    grp["fraction"] = grp["count"] / totals.clip(lower=1) * 100
    grp["lambda2_std_str"] = grp["lambda2_std"].map(L2_LABELS)

    fig = px.bar(
        grp,
        x="lambda2_std_str",
        y="fraction",
        color="morphotype",
        barmode="stack",
        facet_col="wind_regime",
        color_discrete_map=MORPHO_COLORS,
        labels={
            "lambda2_std_str": "Heterogeneidad λ₂",
            "fraction": "% de dunas",
            "morphotype": "Morfotipo",
        },
    )
    apply_style(fig, "C3 — Composición morfológica al final")
    save_fig(fig, "C3_morphotype_composition", out_dir, html)


def fig_C4_lambda2_vs_asymmetry(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """C4: Scatter lambda2 vs asymmetry, coloreado por morphotype."""
    # Muestra aleatoria para no sobrecargar la figura
    sub = agent_final[agent_final["lambda2_std"] > 0].sample(
        min(8000, len(agent_final)), random_state=42
    )
    fig = px.scatter(
        sub,
        x="lambda2",
        y="asymmetry",
        color="morphotype",
        facet_col="wind_regime",
        color_discrete_map=MORPHO_COLORS,
        opacity=0.35,
        labels={
            "lambda2": "λ₂ individual del agente",
            "asymmetry": "Índice de asimetría",
            "morphotype": "Morfotipo",
        },
        render_mode="svg",
    )
    apply_style(fig, "C4 — λ₂ individual vs asimetría (solo corridas heterogéneas)")
    save_fig(fig, "C4_lambda2_vs_asymmetry", out_dir, html)


def fig_C5_lambda2_vs_width(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """C5: Scatter lambda2 vs width."""
    sub = agent_final[agent_final["lambda2_std"] > 0].sample(
        min(8000, len(agent_final)), random_state=42
    )
    fig = px.scatter(
        sub,
        x="lambda2",
        y="width",
        color="morphotype",
        facet_col="wind_regime",
        color_discrete_map=MORPHO_COLORS,
        opacity=0.35,
        labels={
            "lambda2": "λ₂ individual del agente",
            "width": "Ancho total (m)",
            "morphotype": "Morfotipo",
        },
        render_mode="svg",
    )
    apply_style(fig, "C5 — λ₂ individual vs ancho total (solo corridas heterogéneas)")
    save_fig(fig, "C5_lambda2_vs_width", out_dir, html)


def fig_C6_lambda2_survivor(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """C6: Distribución de lambda2 superviviente vs distribución inicial teórica."""
    sub = agent_final[agent_final["lambda2_std"] > 0]
    regimes = sub["wind_regime"].unique()
    for regime in regimes:
        for l2_std in sorted(sub["lambda2_std"].unique()):
            grp = sub[(sub["wind_regime"] == regime) & (sub["lambda2_std"] == l2_std)]
            if len(grp) < 10:
                continue
            l2_mean = 1.8  # FIXED_PARAM del paper
            x_theo = np.linspace(max(1.0, l2_mean - 4*l2_std),
                                  l2_mean + 4*l2_std, 200)
            y_theo = (1 / (l2_std * np.sqrt(2*np.pi)) *
                      np.exp(-0.5*((x_theo - l2_mean)/l2_std)**2))

            fig = go.Figure()
            # Distribución observada (supervivientes)
            fig.add_trace(go.Histogram(
                x=grp["lambda2"],
                histnorm="probability density",
                name="Supervivientes (final)",
                marker_color=L2_COLORS[l2_std],
                opacity=0.65,
                nbinsx=30,
            ))
            # Distribución inicial teórica
            fig.add_trace(go.Scatter(
                x=x_theo, y=y_theo,
                name=f"Inicial teórica N(1.8, {l2_std}²)",
                line=dict(color="#333333", width=2, dash="dash"),
                mode="lines",
            ))
            label = L2_LABELS.get(l2_std, f"λ₂_std={l2_std}")
            apply_style(fig,
                        f"C6 — λ₂ superviviente vs inicial: {regime} | {label}",
                        "λ₂", "Densidad de probabilidad")
            save_fig(fig,
                     f"C6_lambda2_survivor_{regime}_l2std{l2_std:.2f}",
                     out_dir, html)


def fig_C7_spatial_segregation(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """C7: Posición downwind (pos_y) vs lambda2 — segregación espacial emergente."""
    sub = agent_final[agent_final["lambda2_std"] > 0].sample(
        min(8000, len(agent_final)), random_state=42
    )
    fig = px.scatter(
        sub,
        x="lambda2",
        y="pos_y",
        color="width",
        facet_col="wind_regime",
        color_continuous_scale="Viridis",
        opacity=0.40,
        labels={
            "lambda2": "λ₂ individual",
            "pos_y": "Posición sotavento (m)",
            "width": "Ancho (m)",
        },
        render_mode="svg",
    )
    apply_style(fig, "C7 — Segregación espacial: posición vs λ₂ individual")
    save_fig(fig, "C7_spatial_segregation", out_dir, html)


def fig_C8_regime_comparison_violin(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """C8 (bonus): Violinplot de anchos comparando regímenes de viento."""
    df = agent_final.copy()
    df["lambda2_std_str"] = df["lambda2_std"].map(L2_LABELS)
    fig = px.violin(
        df,
        x="lambda2_std_str",
        y="width",
        color="wind_regime",
        box=True,
        points=False,
        color_discrete_map=REGIME_COLORS,
        labels={
            "lambda2_std_str": "Heterogeneidad λ₂",
            "width": "Ancho total (m)",
            "wind_regime": "Régimen de viento",
        },
        category_orders={"lambda2_std_str": list(L2_LABELS.values())},
    )
    apply_style(fig, "C8 — Distribución de anchos: unimodal vs bimodal_acute")
    save_fig(fig, "C8_width_violin_regimes", out_dir, html)


# ═══════════════════════════════════════════════════════════════════════════════
# GRUPO D — Distribuciones y líneas de tiempo enriquecidas
# ═══════════════════════════════════════════════════════════════════════════════

def _kde(data: np.ndarray, x_grid: np.ndarray, bw: float | None = None) -> np.ndarray:
    """KDE gaussiana manual — no requiere scipy."""
    if len(data) == 0:
        return np.zeros_like(x_grid)
    h = bw if bw else 1.06 * np.std(data) * len(data) ** (-0.2)
    h = max(h, 1e-6)
    diffs = (x_grid[:, None] - data[None, :]) / h
    return np.mean(np.exp(-0.5 * diffs**2), axis=1) / (h * np.sqrt(2 * np.pi))


def fig_D1_kde_width_by_qshift(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """D1: KDE de anchos — subplots por qshift_ratio, color por lambda2_std.

    Figura central del resultado: muestra cómo qshift_ratio desplaza el pico
    y lambda2_std ensancha/aplana la distribución de tamaños.
    """
    if agent_final.empty:
        print("  WARN D1: agent_final vacío, se omite")
        return

    qshift_vals = sorted(agent_final["qshift_ratio"].unique())
    n_cols = len(qshift_vals)
    l2_vals = sorted(agent_final["lambda2_std"].unique())

    x_min = max(0, agent_final["width"].quantile(0.01))
    x_max = agent_final["width"].quantile(0.99)
    x_grid = np.linspace(x_min, x_max, 300)

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[f"qshift = {v:.2f}" for v in qshift_vals],
        shared_yaxes=True,
    )
    for col, qshift in enumerate(qshift_vals, 1):
        sub_q = agent_final[agent_final["qshift_ratio"] == qshift]
        for l2_std in l2_vals:
            grp = sub_q[sub_q["lambda2_std"] == l2_std]["width"].dropna().values
            if len(grp) < 5:
                continue
            y = _kde(grp, x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid, y=y,
                    mode="lines",
                    name=L2_LABELS.get(l2_std, f"λ₂_std={l2_std}"),
                    line=dict(color=L2_COLORS.get(l2_std, "#888"), width=2),
                    legendgroup=str(l2_std),
                    showlegend=(col == 1),
                    fill="tozeroy",
                    fillcolor=hex_to_rgba(L2_COLORS.get(l2_std, "#888888"), 0.08),
                ),
                row=1, col=col,
            )
    fig.update_xaxes(title_text="Ancho total (m)")
    fig.update_yaxes(title_text="Densidad de prob.", col=1)
    apply_style(fig, "D1 — KDE de anchos: subplots por qshift_ratio, color por λ₂_std")
    save_fig(fig, "D1_kde_width_qshift", out_dir, html)


def fig_D2_ecdf_width(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """D2: ECDF de anchos — subplots por outflux_mode, color por lambda2_std.

    La separación horizontal entre curvas mide el efecto de la heterogeneidad
    sobre la mediana de tamaños.
    """
    if agent_final.empty:
        print("  WARN D2: agent_final vacío, se omite")
        return

    outflux_vals = sorted(agent_final["outflux_mode"].unique())
    n_cols = len(outflux_vals)
    l2_vals = sorted(agent_final["lambda2_std"].unique())

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[f"outflux: {v}" for v in outflux_vals],
        shared_yaxes=True,
    )
    for col, outflux in enumerate(outflux_vals, 1):
        sub_o = agent_final[agent_final["outflux_mode"] == outflux]
        for l2_std in l2_vals:
            grp = np.sort(sub_o[sub_o["lambda2_std"] == l2_std]["width"].dropna().values)
            if len(grp) < 5:
                continue
            ecdf_y = np.arange(1, len(grp) + 1) / len(grp)
            fig.add_trace(
                go.Scatter(
                    x=grp, y=ecdf_y,
                    mode="lines",
                    name=L2_LABELS.get(l2_std, f"λ₂_std={l2_std}"),
                    line=dict(color=L2_COLORS.get(l2_std, "#888"), width=2),
                    legendgroup=str(l2_std),
                    showlegend=(col == 1),
                ),
                row=1, col=col,
            )
    fig.update_xaxes(title_text="Ancho total (m)")
    fig.update_yaxes(title_text="Probabilidad acumulada", col=1)
    apply_style(fig, "D2 — ECDF de anchos: subplots por outflux_mode, color por λ₂_std")
    save_fig(fig, "D2_ecdf_width_outflux", out_dir, html)


def fig_D3_pairplot(summary: pd.DataFrame, out_dir: Path, html: bool):
    """D3: Scatter matriz reducida — 4 métricas de salida, color por lambda2_std.

    Variables: mean_width_final, cv_width, calving_rate, mean_asymmetry_final.
    Detecta correlaciones entre métricas de salida en función de la heterogeneidad.
    """
    df = filter_viable(summary, "D3")
    df["calving_rate"] = df["calving_count"] / df["n_steps_run"].clip(lower=1)
    df["cv_width"]     = df["std_width_final"] / df["mean_width_final"].clip(lower=0.01)
    df["lambda2_std_str"] = df["lambda2_std"].map(L2_LABELS)

    vars_ = [
        ("mean_width_final",    "Ancho medio final (m)"),
        ("cv_width",            "CV anchos"),
        ("calving_rate",        "Tasa calveo"),
        ("mean_asymmetry_final","Asimetría media"),
    ]
    n = len(vars_)
    fig = make_subplots(rows=n, cols=n,
                        shared_xaxes=False, shared_yaxes=False)

    for row, (var_y, lab_y) in enumerate(vars_, 1):
        for col, (var_x, lab_x) in enumerate(vars_, 1):
            for l2_std in sorted(df["lambda2_std"].unique()):
                grp = df[df["lambda2_std"] == l2_std]
                color = L2_COLORS.get(l2_std, "#888")
                label = L2_LABELS.get(l2_std, f"λ₂_std={l2_std}")
                show = (row == 1 and col == 2)  # leyenda solo una vez
                if row == col:
                    # Diagonal: histograma
                    fig.add_trace(go.Histogram(
                        x=grp[var_x], nbinsx=15,
                        marker_color=color, opacity=0.55,
                        name=label, legendgroup=str(l2_std),
                        showlegend=show,
                    ), row=row, col=col)
                else:
                    fig.add_trace(go.Scatter(
                        x=grp[var_x], y=grp[var_y],
                        mode="markers",
                        marker=dict(color=color, size=5, opacity=0.6),
                        name=label, legendgroup=str(l2_std),
                        showlegend=show,
                    ), row=row, col=col)
            if col == 1:
                fig.update_yaxes(title_text=lab_y, row=row, col=col)
            if row == n:
                fig.update_xaxes(title_text=lab_x, row=row, col=col)

    fig.update_layout(height=800)
    apply_style(fig, "D3 — Scatter matriz: métricas de salida, color por λ₂_std")
    save_fig(fig, "D3_pairplot", out_dir, html)


def fig_D4_correlation_heatmap(summary: pd.DataFrame, out_dir: Path, html: bool):
    """D4: Heatmap de correlación de Pearson — uno por lambda2_std.

    Muestra si la heterogeneidad cambia qué parámetro de entrada controla
    qué métrica de salida.
    """
    df = filter_viable(summary, "D4")
    df["calving_rate"] = df["calving_count"] / df["n_steps_run"].clip(lower=1)
    df["cv_width"]     = df["std_width_final"] / df["mean_width_final"].clip(lower=0.01)

    input_vars  = ["qsat", "q0ratio", "qshift_ratio"]
    output_vars = ["n_dunes_final", "mean_width_final", "cv_width",
                   "calving_rate", "mean_asymmetry_final"]

    l2_vals = sorted(df["lambda2_std"].unique())
    n = len(l2_vals)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[L2_LABELS.get(v, f"λ₂_std={v}") for v in l2_vals],
        shared_yaxes=True,
    )
    for col, l2_std in enumerate(l2_vals, 1):
        sub = df[df["lambda2_std"] == l2_std][input_vars + output_vars].dropna()
        if len(sub) < 4:
            continue
        corr = sub.corr(method="pearson").loc[output_vars, input_vars]
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=input_vars,
                y=output_vars,
                colorscale="RdBu",
                zmid=0, zmin=-1, zmax=1,
                showscale=(col == n),
                colorbar=dict(title="r", len=0.8),
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=11),
            ),
            row=1, col=col,
        )
    apply_style(fig, "D4 — Correlación de Pearson: entradas vs salidas, por λ₂_std")
    save_fig(fig, "D4_correlation_heatmap", out_dir, html)


def fig_D5_width_percentile_band(model_data: pd.DataFrame,
                                  agent_final: pd.DataFrame,
                                  out_dir: Path, html: bool):
    """D5: Evolución mean_width con banda p10–p90 — subplots por qsat, color por lambda2_std.

    La banda no es ±std entre réplicas sino p10–p90 de la distribución de agentes.
    Requiere agent_data con columna 'step' para el cruce.
    """
    if agent_final.empty:
        print("  WARN D5: agent_final vacío, se omite")
        return

    step_col      = require_col(model_data, "step",       "D5")
    mwidth_col    = require_col(model_data, "mean_width", "D5")

    qsat_vals = sorted(model_data["qsat"].unique())
    n_cols    = len(qsat_vals)
    l2_vals   = sorted(model_data["lambda2_std"].unique())

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[f"qsat = {v:.0f}" for v in qsat_vals],
        shared_yaxes=True,
    )
    for col, qsat in enumerate(qsat_vals, 1):
        sub = model_data[model_data["qsat"] == qsat]
        for l2_std in l2_vals:
            grp = sub[sub["lambda2_std"] == l2_std].copy()
            if grp.empty:
                continue
            max_s = grp[step_col].max()
            grp["step_norm"] = grp[step_col] / max(max_s, 1)
            grp["step_bin"]  = pd.cut(grp["step_norm"], bins=40, labels=False)

            ts = grp.groupby("step_bin")[mwidth_col].agg(
                mean="mean", p10=lambda x: x.quantile(0.10),
                p90=lambda x: x.quantile(0.90)
            ).reset_index()
            x     = ts["step_bin"] / 40
            color = L2_COLORS.get(l2_std, "#888888")
            label = L2_LABELS.get(l2_std, f"λ₂_std={l2_std}")

            # Banda p10–p90
            fig.add_trace(go.Scatter(
                x=pd.concat([x, x[::-1]]),
                y=pd.concat([ts["p90"], ts["p10"][::-1]]),
                fill="toself",
                fillcolor=hex_to_rgba(color, 0.12),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col)
            # Línea media
            fig.add_trace(go.Scatter(
                x=x, y=ts["mean"],
                mode="lines",
                name=label, legendgroup=str(l2_std),
                line=dict(color=color, width=2),
                showlegend=(col == 1),
            ), row=1, col=col)

    fig.update_xaxes(title_text="Fracción del tiempo simulado")
    fig.update_yaxes(title_text="Ancho medio (m)", col=1)
    apply_style(fig, "D5 — Ancho medio con banda p10–p90: subplots por qsat, color por λ₂_std")
    save_fig(fig, "D5_width_percentile_band", out_dir, html)


def fig_D6_bumpchart_collisions(summary: pd.DataFrame, out_dir: Path, html: bool):
    """D6: Bumpchart de tipos de colisión vs qshift_ratio.

    Subplots por outflux_mode, color por lambda2_std. Líneas conectando los
    3 valores de qshift_ratio para cada lambda2_std. Responde la hipótesis del
    paper 2024 sobre el punto de equilibrio fusión/fragmentación.
    """
    df = filter_viable(summary, "D6")
    total = df["collision_count"].clip(lower=1)
    df["frac_merging"]       = df["merging_count"]       / total
    df["frac_fragmentation"] = df["fragmentation_count"] / total
    df["frac_exchange"]      = df["exchange_count"]       / total
    df["lambda2_std_str"]    = df["lambda2_std"].map(L2_LABELS)

    outflux_vals = sorted(df["outflux_mode"].unique())
    metrics = [
        ("frac_merging",       "Fusión",         "solid"),
        ("frac_exchange",      "Intercambio",    "dot"),
        ("frac_fragmentation", "Fragmentación",  "dash"),
    ]

    fig = make_subplots(
        rows=len(metrics), cols=len(outflux_vals),
        subplot_titles=[f"{m[1]} — {o}" for m in metrics for o in outflux_vals],
        shared_xaxes=True, shared_yaxes="rows",
    )
    for row, (metric, label, dash) in enumerate(metrics, 1):
        for col, outflux in enumerate(outflux_vals, 1):
            sub = df[df["outflux_mode"] == outflux]
            agg = sub.groupby(["lambda2_std", "qshift_ratio"])[metric].mean().reset_index()
            for l2_std in sorted(agg["lambda2_std"].unique()):
                grp = agg[agg["lambda2_std"] == l2_std].sort_values("qshift_ratio")
                color = L2_COLORS.get(l2_std, "#888")
                lbl   = L2_LABELS.get(l2_std, f"λ₂_std={l2_std}")
                fig.add_trace(go.Scatter(
                    x=grp["qshift_ratio"].astype(str),
                    y=grp[metric] * 100,
                    mode="lines+markers",
                    name=lbl, legendgroup=str(l2_std),
                    showlegend=(row == 1 and col == 1),
                    line=dict(color=color, width=2, dash=dash),
                    marker=dict(size=7),
                ), row=row, col=col)
        fig.update_yaxes(title_text=f"{label} (%)", row=row, col=1)

    fig.update_xaxes(title_text="qshift_ratio", row=len(metrics))
    fig.update_layout(height=700)
    apply_style(fig, "D6 — Bumpchart: fracción de colisiones vs qshift_ratio")
    save_fig(fig, "D6_bumpchart_collisions", out_dir, html)


def fig_D7_strip_lambda2(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """D7: Strip plot de lambda2 superviviente — subplots por wind_regime.

    Cada punto es un agente. Jitter horizontal para legibilidad.
    Si el centro de masa está desplazado de 1.8 (lambda2_mean), hay selección emergente.
    """
    sub = agent_final[agent_final["lambda2_std"] > 0].copy()
    if sub.empty:
        print("  WARN D7: sin datos heterogéneos, se omite")
        return

    regimes = sorted(sub["wind_regime"].unique())
    n_cols  = len(regimes)

    # Muestra para no saturar
    sub = sub.sample(min(6000, len(sub)), random_state=42)
    sub["lambda2_std_str"] = sub["lambda2_std"].map(L2_LABELS)

    # Jitter
    rng = np.random.default_rng(0)
    sub["x_jitter"] = (sub["lambda2_std"].rank(method="dense") - 1
                       + rng.uniform(-0.25, 0.25, len(sub)))

    fig = make_subplots(rows=1, cols=n_cols,
                        subplot_titles=[f"Viento: {r}" for r in regimes],
                        shared_yaxes=True)
    for col, regime in enumerate(regimes, 1):
        grp = sub[sub["wind_regime"] == regime]
        for l2_std in sorted(grp["lambda2_std"].unique()):
            sg = grp[grp["lambda2_std"] == l2_std]
            fig.add_trace(go.Scatter(
                x=sg["x_jitter"],
                y=sg["lambda2"],
                mode="markers",
                marker=dict(color=L2_COLORS.get(l2_std, "#888"),
                            size=3, opacity=0.4),
                name=L2_LABELS.get(l2_std, f"λ₂_std={l2_std}"),
                legendgroup=str(l2_std),
                showlegend=(col == 1),
            ), row=1, col=col)
            # Línea de mediana
            med = sg["lambda2"].median()
            x_center = sg["x_jitter"].mean()
            fig.add_trace(go.Scatter(
                x=[x_center - 0.3, x_center + 0.3],
                y=[med, med],
                mode="lines",
                line=dict(color=L2_COLORS.get(l2_std, "#888"), width=3),
                showlegend=False,
            ), row=1, col=col)
        # Línea de referencia lambda2_mean = 1.8
        fig.add_hline(y=1.8, line_dash="dash", line_color="#333333",
                      annotation_text="λ₂_mean=1.8", row=1, col=col)

    fig.update_xaxes(showticklabels=False, title_text="λ₂_std (jitter horizontal)")
    fig.update_yaxes(title_text="λ₂ individual superviviente", col=1)
    apply_style(fig, "D7 — Strip plot: λ₂ superviviente por régimen (mediana y referencia 1.8)")
    save_fig(fig, "D7_strip_lambda2", out_dir, html)


def fig_D8_hexbin_position_width(agent_final: pd.DataFrame, out_dir: Path, html: bool):
    """D8: Mapa de calor 2D posición × ancho — subplots por lambda2_std.

    Muestra si las dunas grandes se acumulan en ciertas zonas del campo
    (gradiente downwind de tamaño, coherente con paper 2024 Sección 3.1.2).
    """
    if agent_final.empty:
        print("  WARN D8: agent_final vacío, se omite")
        return

    l2_vals = sorted(agent_final["lambda2_std"].unique())
    n_cols  = len(l2_vals)

    w_max   = agent_final["width"].quantile(0.98)
    pos_min = agent_final["pos_y"].quantile(0.01)
    pos_max = agent_final["pos_y"].quantile(0.99)

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[L2_LABELS.get(v, f"λ₂_std={v}") for v in l2_vals],
        shared_yaxes=True, shared_xaxes=True,
    )
    for col, l2_std in enumerate(l2_vals, 1):
        grp = agent_final[agent_final["lambda2_std"] == l2_std]
        fig.add_trace(
            go.Histogram2d(
                x=grp["width"].clip(upper=w_max),
                y=grp["pos_y"].clip(lower=pos_min, upper=pos_max),
                nbinsx=30, nbinsy=30,
                colorscale="Viridis",
                showscale=(col == n_cols),
                colorbar=dict(title="N agentes", len=0.8),
            ),
            row=1, col=col,
        )
    fig.update_xaxes(title_text="Ancho (m)", row=1)
    fig.update_yaxes(title_text="Posición downwind (m)", col=1)
    apply_style(fig, "D8 — Densidad 2D: posición downwind × ancho, por λ₂_std")
    save_fig(fig, "D8_hexbin_position_width", out_dir, html)


def fig_D9_morphotype_area(model_data: pd.DataFrame, out_dir: Path, html: bool):
    """D9: Área apilada de morfotipos vs tiempo — subplots por lambda2_std.

    Requiere columnas de conteo por morfotipo en model_data. Si no existen,
    reconstruye desde N_dunes y mean_asymmetry como proxy.
    """
    df = model_data.copy()
    step_col = require_col(df, "step", "D9")

    # Detectar si hay columnas de morfotipo
    morpho_cols = {
        "N_barchan":    ["n_barchan",    "N_barchan"],
        "N_transverse": ["n_transverse", "N_transverse"],
        "N_asymmetric": ["n_asymmetric", "N_asymmetric"],
        "N_pre_calving":["n_pre_calving","N_pre_calving"],
    }
    available = {}
    for canonical, aliases in morpho_cols.items():
        for a in aliases:
            if a in df.columns or a.lower() in [c.lower() for c in df.columns]:
                real = next(c for c in df.columns if c.lower() == a.lower())
                available[canonical] = real
                break

    if not available:
        print("  WARN D9: no hay columnas de morfotipo en model_data. "
              "Añade N_barchan/N_transverse/N_asymmetric/N_pre_calving al DataCollector "
              "para generar esta figura.")
        return

    morpho_colors_area = {
        "N_barchan":     "#2166ac",
        "N_transverse":  "#4dac26",
        "N_asymmetric":  "#e07b39",
        "N_pre_calving": "#d6604d",
    }

    max_step = df.groupby("run_id")[step_col].transform("max")
    df["step_norm"] = df[step_col] / max_step.clip(lower=1)
    df["step_bin"]  = pd.cut(df["step_norm"], bins=50, labels=False)

    regimes = df["wind_regime"].unique()
    for regime in regimes:
        sub_r = df[df["wind_regime"] == regime]
        l2_vals = sorted(sub_r["lambda2_std"].unique())
        n_cols  = len(l2_vals)
        fig = make_subplots(
            rows=1, cols=n_cols,
            subplot_titles=[L2_LABELS.get(v, f"λ₂_std={v}") for v in l2_vals],
            shared_yaxes=True,
        )
        for col, l2_std in enumerate(l2_vals, 1):
            grp = sub_r[sub_r["lambda2_std"] == l2_std]
            ts  = grp.groupby("step_bin")[[v for v in available.values()]].mean().reset_index()
            x   = ts["step_bin"] / 50
            for canonical, real_col in available.items():
                fig.add_trace(go.Scatter(
                    x=x, y=ts[real_col],
                    mode="lines",
                    stackgroup="one",
                    name=canonical.replace("N_", ""),
                    legendgroup=canonical,
                    showlegend=(col == 1),
                    line=dict(width=0),
                    fillcolor=morpho_colors_area.get(canonical, "#888"),
                ), row=1, col=col)
        fig.update_xaxes(title_text="Fracción del tiempo simulado")
        fig.update_yaxes(title_text="N dunas por morfotipo", col=1)
        apply_style(fig, f"D9 — Evolución morfotipos (área apilada): {regime}")
        save_fig(fig, f"D9_morphotype_area_{regime}", out_dir, html)


def fig_D10_calving_rate_vs_qsat(summary: pd.DataFrame, out_dir: Path, html: bool):
    """D10 (bonus): Líneas de tasa de calveo vs qsat — subplots por wind_regime,
    color por lambda2_std, estilo de línea por outflux_mode.

    Complementa D6: muestra si la intensidad del forzamiento (qsat) interactúa
    con la heterogeneidad para modular los calveos.
    """
    df = filter_viable(summary, "D10")
    df["calving_rate"] = df["calving_count"] / df["n_steps_run"].clip(lower=1)

    regimes = sorted(df["wind_regime"].unique())
    n_cols  = len(regimes)
    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[f"Viento: {r}" for r in regimes],
        shared_yaxes=True,
    )
    for col, regime in enumerate(regimes, 1):
        sub = df[df["wind_regime"] == regime]
        for outflux in sorted(sub["outflux_mode"].unique()):
            for l2_std in sorted(sub["lambda2_std"].unique()):
                grp = (sub[(sub["outflux_mode"] == outflux) &
                           (sub["lambda2_std"] == l2_std)]
                       .groupby("qsat")["calving_rate"].mean()
                       .reset_index())
                color = L2_COLORS.get(l2_std, "#888")
                dash  = OUTFLUX_DASH.get(outflux, "solid")
                label = f"{L2_LABELS.get(l2_std, str(l2_std))} | {outflux}"
                fig.add_trace(go.Scatter(
                    x=grp["qsat"], y=grp["calving_rate"],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2, dash=dash),
                    marker=dict(size=7),
                    legendgroup=label,
                    showlegend=(col == 1),
                ), row=1, col=col)
    fig.update_xaxes(title_text="qsat (m²/año)")
    fig.update_yaxes(title_text="Tasa de calveo (calveos/paso)", col=1)
    apply_style(fig, "D10 — Tasa de calveo vs qsat: color por λ₂_std, estilo por outflux")
    save_fig(fig, "D10_calving_rate_vs_qsat", out_dir, html)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Análisis de sensibilidad ABM dunas")
    parser.add_argument("--data", default="resultados/",
                        help="Directorio con summary.parquet y dunas.db")
    parser.add_argument("--out", default="figures/",
                        help="Directorio de salida para las figuras")
    parser.add_argument("--no-html", action="store_true",
                        help="Solo generar PNG, no HTML interactivo")
    parser.add_argument("--skip-agent", action="store_true",
                        help="Omitir Grupo C (agent_data, lento para DBs grandes)")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = not args.no_html

    print(f"\n{'='*60}")
    print(f"  ANÁLISIS DE SENSIBILIDAD — ABM DUNAS BARCHÁN")
    print(f"  Datos  : {data_dir.resolve()}")
    print(f"  Figuras: {out_dir.resolve()}")
    print(f"{'='*60}\n")

    # ── Carga ────────────────────────────────────────────────────────────────
    print("Cargando summary.parquet...", flush=True)
    summary = load_summary(data_dir)
    print(f"  {len(summary)} corridas | regímenes: {summary['wind_regime'].unique().tolist()}")

    print("Cargando model_data desde SQLite...", flush=True)
    model_data = load_model_data(data_dir)
    print(f"  {len(model_data)} filas")
    print(f"  Columnas disponibles: {list(model_data.columns)}")

    if not args.skip_agent:
        print("Cargando agent_data (último 20% de pasos)...", flush=True)
        agent_final = load_agent_data(data_dir, last_frac=0.20)
        print(f"  {len(agent_final)} filas de agentes")
    else:
        agent_final = pd.DataFrame()

    # ── Grupo A ──────────────────────────────────────────────────────────────
    print("\n── Grupo A: Heatmaps de summary ──────────────────────────────")
    fig_A1_heatmap_calving(summary, out_dir, html)
    fig_A2_heatmap_cv_width(summary, out_dir, html)
    fig_A3_heatmap_collisions(summary, out_dir, html)
    fig_A4_scatter_ndunes_calving(summary, out_dir, html)
    fig_A5_heatmap_collision_fraction(summary, out_dir, html)

    # ── Grupo B ──────────────────────────────────────────────────────────────
    print("\n── Grupo B: Series temporales ────────────────────────────────")
    fig_B1_ma_calvings(model_data, out_dir, html)
    fig_B2_ndunes_evolution(model_data, out_dir, html)
    fig_B3_width_evolution(model_data, out_dir, html)
    fig_B4_collision_breakdown(model_data, out_dir, html)
    fig_B5_collision_fractions_bar(summary, out_dir, html)

    # ── Grupo C ──────────────────────────────────────────────────────────────
    if not args.skip_agent and len(agent_final) > 0:
        print("\n── Grupo C: Estado final de agentes ──────────────────────────")
        fig_C1_width_distribution(agent_final, out_dir, html)
        fig_C2_asymmetry_distribution(agent_final, out_dir, html)
        fig_C3_morphotype_composition(agent_final, out_dir, html)
        fig_C4_lambda2_vs_asymmetry(agent_final, out_dir, html)
        fig_C5_lambda2_vs_width(agent_final, out_dir, html)
        fig_C6_lambda2_survivor(agent_final, out_dir, html)
        fig_C7_spatial_segregation(agent_final, out_dir, html)
        fig_C8_regime_comparison_violin(agent_final, out_dir, html)
    else:
        print("\n  (Grupo C omitido — usa --skip-agent=False para activarlo)")

    # ── Grupo D ──────────────────────────────────────────────────────────────
    print("\n── Grupo D: Distribuciones y líneas de tiempo enriquecidas ───")
    # D1, D2, D5, D7, D8 requieren agent_data
    if not args.skip_agent and len(agent_final) > 0:
        fig_D1_kde_width_by_qshift(agent_final, out_dir, html)
        fig_D2_ecdf_width(agent_final, out_dir, html)
        fig_D7_strip_lambda2(agent_final, out_dir, html)
        fig_D8_hexbin_position_width(agent_final, out_dir, html)
        fig_D5_width_percentile_band(model_data, agent_final, out_dir, html)
    else:
        print("  (D1, D2, D5, D7, D8 omitidas — requieren agent_data)")
    # D3, D4, D6, D9, D10 solo necesitan summary o model_data
    fig_D3_pairplot(summary, out_dir, html)
    fig_D4_correlation_heatmap(summary, out_dir, html)
    fig_D6_bumpchart_collisions(summary, out_dir, html)
    fig_D9_morphotype_area(model_data, out_dir, html)
    fig_D10_calving_rate_vs_qsat(summary, out_dir, html)

    # ── Resumen ──────────────────────────────────────────────────────────────
    pngs = list(out_dir.glob("*.png"))
    print(f"\n{'='*60}")
    print(f"  {len(pngs)} figuras generadas en {out_dir.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()