"""
visualization/shared/callbacks.py
Funciones auxiliares compartidas entre stored_results y real_time.

Responsabilidades
-----------------
1. Carga de datos     — summary, run, agentes por paso
2. Figuras Plotly     — campo de dunas, series de tiempo, histograma,
                        heatmap de parámetros, coordenadas paralelas
3. Sin layout Dash    — este módulo no importa dash, solo plotly + pandas

Notas importantes
-----------------
- make_field_figure ahora soporta field_view:
    "domain" : muestra todo el dominio de simulación.
    "active" : muestra la franja activa definida por fieldwidth.
    "auto"   : ajusta la vista a las dunas presentes en el paso actual.

- Se usan constrain="domain", scaleanchor="x" y scaleratio=1 para reducir
  distorsiones visuales y mantener escala métrica consistente.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from shapely.affinity import rotate as shapely_rotate
    from shapely.affinity import translate as shapely_translate
    from shapely.geometry import Polygon
    _SHAPELY = True
except ImportError:
    _SHAPELY = False


# ── Constantes compartidas ────────────────────────────────────────────────────

METRIC_LABELS = {
    "n_dunes_final":        "N dunas (final)",
    "mean_width_final":     "Ancho medio (m)",
    "mean_asymmetry_final": "Asimetría media",
    "calving_rate":         "Calveos / paso",
    "collision_rate":       "Colisiones / paso",
    "p90_width_final":      "P90 ancho (m)",
    "calving_count":        "Calveos totales",
    "collision_count":      "Colisiones totales",
}

PARAM_LABELS = {
    "qsat":         "q_sat (m²/año)",
    "q0ratio":      "q₀ / q_sat",
    "qshift_ratio": "q_shift / q_sat",
    "lambda2_std":  "λ₂ σ heterogeneidad",
    "lambda2_mean": "λ₂ media",
    "n_dunes_init": "N dunas iniciales",
}

ALL_REGIMES = [
    "unimodal",
    "bimodal_acute",
    "bimodal_moderate",
    "bimodal_obtuse",
    "multidirectional",
]

MORPHOTYPE_COLORS = {
    "barchan":     "#4A90D9",
    "transverse":  "#38A169",
    "asymmetric":  "#DD6B20",
    "pre_calving": "#E53E3E",
    "ghost":       "#A0AEC0",
}

_CONTINUOUS_SCALES = {
    "lambda2":   "Viridis",
    "asymmetry": "RdBu",
    "width":     "Plasma",
}

C = {
    "bg":     "#F4F5F7",
    "card":   "#FFFFFF",
    "border": "#E2E5EA",
    "text":   "#1A202C",
    "muted":  "#718096",
    "accent": "#4A90D9",
    "warn":   "#DD6B20",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, Arial", size=11, color=C["text"]),
)


# ── 1. Carga de datos ─────────────────────────────────────────────────────────

def load_summary(data_dir: Path) -> pd.DataFrame:
    """Carga summary.parquet o run_index.csv. Retorna DataFrame vacío si no existe."""
    for p in [data_dir / "summary.parquet", data_dir / "run_index.csv"]:
        if p.exists():
            try:
                return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            except Exception:
                continue
    return pd.DataFrame()


def load_run(data_dir: Path, run_id: str) -> dict:
    """
    Carga params, model_data y agent_data de una corrida.

    Retorna
    -------
    dict:
        params : dict
        model  : pd.DataFrame
        agents : pd.DataFrame
    """
    run_dir = data_dir / "runs" / run_id

    params = {}
    params_path = run_dir / "params.json"
    if params_path.exists():
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)

    model_df = pd.DataFrame()
    model_path = run_dir / "model_data.parquet"
    if model_path.exists():
        model_df = pd.read_parquet(model_path)

    agent_df = pd.DataFrame()
    agent_path = run_dir / "agent_data.parquet"
    if agent_path.exists():
        agent_df = pd.read_parquet(agent_path)

    return {
        "params": params,
        "model": model_df,
        "agents": agent_df,
    }


def get_steps(agent_df: pd.DataFrame) -> list[int]:
    """Retorna lista ordenada de pasos disponibles en agent_data."""
    if agent_df.empty:
        return []

    idx = agent_df.index

    if isinstance(idx, pd.MultiIndex):
        if "Step" in idx.names:
            return sorted(idx.get_level_values("Step").unique().tolist())
        return sorted(idx.get_level_values(0).unique().tolist())

    if "Step" in agent_df.columns:
        return sorted(agent_df["Step"].unique().tolist())

    return sorted(idx.unique().tolist())


def agents_at_step(agent_df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Extrae el DataFrame de agentes para un paso específico."""
    if agent_df.empty or step is None:
        return pd.DataFrame()

    if isinstance(agent_df.index, pd.MultiIndex):
        try:
            if "Step" in agent_df.index.names:
                return agent_df.xs(step, level="Step")
            return agent_df.xs(step, level=0)
        except KeyError:
            return pd.DataFrame()

    if "Step" in agent_df.columns:
        return agent_df[agent_df["Step"] == step].copy()

    return agent_df[agent_df.index == step].copy()


# ── 2. Viento ─────────────────────────────────────────────────────────────────

def estimate_wind_vec(wind_regime: str) -> tuple[float, float]:
    """
    Estima vector de viento dominante desde el nombre del régimen.

    Convención del modelo:
    - Viento primario 270° → vector (0, -1)
    - Las dunas migran en dirección del vector usado por el modelo.
    """
    vectors = {
        "unimodal":           (0.0, -1.0),
        "bimodal_acute":      (-0.383, -0.924),
        "bimodal_moderate":   (-0.707, -0.707),
        "bimodal_obtuse":     (-0.924, -0.383),
        "multidirectional":   (0.0, -1.0),
        "bimodal":            (0.0, -1.0),
    }
    return vectors.get(wind_regime, (0.0, -1.0))


def _wind_vec_from_params(params: dict) -> tuple[float, float]:
    """
    Extrae vector de viento en orden de prioridad:
      1. wind_mean_deg
      2. wind.mean_deg
      3. wind_vec
      4. wind_regime
      5. wind.regime
    """
    deg = params.get("wind_mean_deg")

    if deg is None and isinstance(params.get("wind"), dict):
        deg = params["wind"].get("mean_deg")

    if deg is not None:
        rad = np.radians(float(deg))
        return float(np.cos(rad)), float(np.sin(rad))

    vec = params.get("wind_vec")
    if vec is not None:
        return float(vec[0]), float(vec[1])

    regime = params.get("wind_regime")
    if regime is None and isinstance(params.get("wind"), dict):
        regime = params["wind"].get("regime")

    return estimate_wind_vec(regime or "unimodal")


# ── 3. Helpers de color ───────────────────────────────────────────────────────

def _add_alpha_to_color(color: str, alpha: float) -> str:
    """Convierte '#rrggbb' o 'rgb(r,g,b)' a 'rgba(r,g,b,alpha)'."""
    if not isinstance(color, str):
        return color

    if color.startswith("#"):
        h = color.lstrip("#")
        if len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

    if color.startswith("rgb("):
        inner = color[4:-1]
        return f"rgba({inner},{alpha})"

    if color.startswith("rgba("):
        return color

    return color


def _assign_agent_colors(
    agent_step: pd.DataFrame,
    color_by: str,
    n_bins: int = 7,
) -> pd.Series:
    """
    Retorna Series de colores alineada con agent_step.index.

    color_by:
      - morphotype
      - lambda2
      - asymmetry
      - width
    """
    fallback = MORPHOTYPE_COLORS["ghost"]

    if agent_step.empty:
        return pd.Series(dtype=object)

    if color_by == "morphotype":
        if "morphotype" not in agent_step.columns:
            return pd.Series(fallback, index=agent_step.index)
        return agent_step["morphotype"].map(MORPHOTYPE_COLORS).fillna(fallback)

    col = color_by

    if col == "width" and "width" not in agent_step.columns:
        if {"lw", "rw"}.issubset(agent_step.columns):
            vals = agent_step["lw"].astype(float) + agent_step["rw"].astype(float)
        else:
            return pd.Series(fallback, index=agent_step.index)
    else:
        if col not in agent_step.columns:
            return pd.Series(fallback, index=agent_step.index)
        vals = agent_step[col].astype(float)

    vals = vals.replace([np.inf, -np.inf], np.nan)

    if vals.dropna().empty:
        return pd.Series(fallback, index=agent_step.index)

    vmin = vals.min()
    vmax = vals.max()

    scale_name = _CONTINUOUS_SCALES.get(col, "Viridis")
    palette = pc.sample_colorscale(scale_name, n_bins)

    if vmax == vmin:
        return pd.Series(palette[n_bins // 2], index=agent_step.index)

    norm = ((vals - vmin) / (vmax - vmin)).clip(0, 1).fillna(0.5)
    bins = (norm * (n_bins - 1)).round().astype(int).clip(0, n_bins - 1)

    return bins.map(lambda i: palette[int(i)])


# ── 4. Geometría de polígonos ─────────────────────────────────────────────────

def _flank_world_coords(
    x: float,
    y: float,
    lw: float,
    rw: float,
    lambda1: float,
    lambda2: float,
    alpha: float,
    delta: float,
    wind_vec: tuple,
    side: str,
    scale: float = 1.0,
) -> tuple[list, list] | None:
    """
    Calcula coordenadas mundo del polígono de un flanco.

    Geometría canónica:
    - Toe en (0, 0)
    - Duna apunta hacia +y
    - Luego se rota según wind_vec y se traslada a (x, y)
    """
    if not _SHAPELY:
        return None

    lw_s = float(lw) * scale
    rw_s = float(rw) * scale

    w = lw_s if side == "left" else rw_s
    sign = -1 if side == "left" else +1

    if w <= 0:
        return None

    d_wide = float(lambda1) * (lw_s + rw_s) / 2.0
    d_horn = float(lambda2) * w

    H = max(1.0, float(alpha) * w + float(delta) / 2.0 * scale)
    H = min(H, w * 0.95)

    pts = [
        (0,              0),
        (sign * w * 0.6, d_wide * 0.3),
        (sign * w,       d_wide),
        (sign * w,       d_horn),
        (sign * H,       d_horn),
        (0,              d_horn * 0.5),
    ]

    poly = Polygon(pts)

    wx, wy = wind_vec
    theta_deg = float(np.rad2deg(np.arctan2(wy, wx)))
    rotation_deg = theta_deg - 90.0

    poly = shapely_rotate(poly, rotation_deg, origin=(0, 0))
    poly = shapely_translate(poly, xoff=x, yoff=y)

    coords = np.array(poly.exterior.coords)

    return (
        coords[:, 0].tolist() + [coords[0, 0]],
        coords[:, 1].tolist() + [coords[0, 1]],
    )


def _get_width_series(agent_step: pd.DataFrame) -> pd.Series:
    """Obtiene ancho total W = lw + rw de forma robusta."""
    if agent_step.empty:
        return pd.Series(dtype=float)

    if "width" in agent_step.columns:
        return agent_step["width"].astype(float)

    if {"lw", "rw"}.issubset(agent_step.columns):
        return agent_step["lw"].astype(float) + agent_step["rw"].astype(float)

    return pd.Series(dtype=float)


def _compute_field_ranges(
    agent_step: pd.DataFrame,
    simwidth: float,
    simlength: float,
    fieldwidth: float,
    field_view: str,
) -> tuple[list[float], list[float]]:
    """
    Calcula rangos X/Y de visualización según field_view.

    field_view:
      - domain
      - active
      - auto
    """
    field_view = field_view or "domain"

    x_range = [0.0, float(simwidth)]
    y_range = [0.0, float(simlength)]

    if field_view == "active":
        x0 = (simwidth - fieldwidth) / 2.0
        x1 = x0 + fieldwidth
        x_range = [x0, x1]
        return x_range, y_range

    if field_view == "auto" and not agent_step.empty:
        if "pos_x" not in agent_step.columns or "pos_y" not in agent_step.columns:
            return x_range, y_range

        xs = agent_step["pos_x"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        ys = agent_step["pos_y"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()

        if xs.empty or ys.empty:
            return x_range, y_range

        widths = _get_width_series(agent_step)
        if not widths.empty:
            typical_w = float(widths.median())
        else:
            typical_w = 50.0

        pad_x = max(100.0, 2.5 * typical_w, 0.10 * (xs.max() - xs.min() + 1))
        pad_y = max(100.0, 2.5 * typical_w, 0.10 * (ys.max() - ys.min() + 1))

        x_range = [
            max(0.0, float(xs.min() - pad_x)),
            min(float(simwidth), float(xs.max() + pad_x)),
        ]
        y_range = [
            max(0.0, float(ys.min() - pad_y)),
            min(float(simlength), float(ys.max() + pad_y)),
        ]

        if x_range[0] == x_range[1]:
            x_range = [
                max(0.0, x_range[0] - 100.0),
                min(float(simwidth), x_range[1] + 100.0),
            ]

        if y_range[0] == y_range[1]:
            y_range = [
                max(0.0, y_range[0] - 100.0),
                min(float(simlength), y_range[1] + 100.0),
            ]

    return x_range, y_range


# ── 5. Figura del campo de dunas ──────────────────────────────────────────────

def make_field_figure(
    agent_step: pd.DataFrame,
    params: dict,
    color_by: str = "flanks",
    step: int = None,
    scale: float = 1.0,
    field_view: str = "domain",
    uirevision_key: str = None,
) -> go.Figure:
    """
    Figura Plotly del campo de dunas.

    field_view:
      - "domain": muestra todo el dominio.
      - "active": muestra solo la franja activa central.
      - "auto": ajusta a las dunas presentes.
    """
    params = params or {}

    simwidth = float(params.get("simwidth", 2000))
    simlength = float(params.get("simlength", 3000))
    fieldwidth = float(params.get("fieldwidth", simwidth))

    wind_vec = _wind_vec_from_params(params)

    lambda1 = float(params.get("lambda1", 1.0))
    alpha = float(params.get("alpha", 0.05))
    delta = float(params.get("delta", 4.6))

    x_range, y_range = _compute_field_ranges(
        agent_step=agent_step,
        simwidth=simwidth,
        simlength=simlength,
        fieldwidth=fieldwidth,
        field_view=field_view,
    )

    # Auto-escala visual de polígonos.
    # Solo agranda el dibujo de las dunas, no sus posiciones.
    if scale <= 1.0 and not agent_step.empty:
        widths = _get_width_series(agent_step)
        if not widths.empty:
            typical_w = float(widths.median())
        else:
            typical_w = 20.0

        visible_span = min(
            max(x_range[1] - x_range[0], 1.0),
            max(y_range[1] - y_range[0], 1.0),
        )

        scale = max(1.0, (visible_span * 0.025) / max(typical_w, 1.0))

    fig = go.Figure()

    fig.update_layout(
        **PLOTLY_LAYOUT,
        uirevision=uirevision_key or "default",
        xaxis=dict(
            range=x_range,
            showgrid=False,
            zeroline=False,
            title="x (m)",
            constrain="domain",
        ),
        yaxis=dict(
            range=y_range,
            showgrid=False,
            zeroline=False,
            title="y (m)",
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=C["border"],
            borderwidth=1,
        ),
        height=420,
        title=dict(
            text=f"Campo de dunas — paso {step}" if step is not None else "Campo de dunas",
            font=dict(size=12),
            x=0.5,
        ),
        margin=dict(l=45, r=20, t=45, b=45),
    )

    # Rectángulo de región activa.
    if fieldwidth < simwidth:
        x0_f = (simwidth - fieldwidth) / 2.0
        fig.add_shape(
            type="rect",
            x0=x0_f,
            x1=x0_f + fieldwidth,
            y0=0,
            y1=simlength,
            line=dict(color=C["border"], width=1, dash="dot"),
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        )

    if agent_step.empty:
        fig.add_annotation(
            text="Sin agentes en este paso",
            x=(x_range[0] + x_range[1]) / 2.0,
            y=(y_range[0] + y_range[1]) / 2.0,
            showarrow=False,
            font=dict(color=C["muted"]),
        )
        _add_wind_arrow(fig, wind_vec, x_range, y_range)
        return fig

    hover_x = []
    hover_y = []
    hover_txt = []

    for _, row in agent_step.iterrows():
        x_ = float(row.get("pos_x", 0))
        y_ = float(row.get("pos_y", 0))
        lw_ = float(row.get("lw", 5))
        rw_ = float(row.get("rw", 5))
        l2_ = float(row.get("lambda2", params.get("lambda2_mean", 1.8)))
        morph = str(row.get("morphotype", "barchan"))
        asym = float(row.get("asymmetry", 0))

        hover_x.append(x_)
        hover_y.append(y_)
        hover_txt.append(
            f"<b>{morph}</b><br>"
            f"W={lw_ + rw_:.1f} m  lw={lw_:.1f}  rw={rw_:.1f}<br>"
            f"λ₂={l2_:.2f}  asim={asym:.3f}<br>"
            f"pos=({x_:.0f}, {y_:.0f})"
        )

    if color_by == "flanks":
        _render_flanks_mode(
            fig=fig,
            agent_step=agent_step,
            lambda1=lambda1,
            alpha=alpha,
            delta=delta,
            wind_vec=wind_vec,
            scale=scale,
            params=params,
        )
    else:
        _render_colorby_mode(
            fig=fig,
            agent_step=agent_step,
            color_by=color_by,
            lambda1=lambda1,
            alpha=alpha,
            delta=delta,
            wind_vec=wind_vec,
            scale=scale,
            params=params,
        )

    # Hover invisible.
    if hover_x:
        fig.add_trace(go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(
                size=max(6 * min(scale, 5), 4),
                color="rgba(0,0,0,0)",
                line=dict(width=0),
            ),
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
            text=hover_txt,
        ))

    _add_wind_arrow(fig, wind_vec, x_range, y_range)

    return fig


def _render_flanks_mode(
    fig,
    agent_step,
    lambda1,
    alpha,
    delta,
    wind_vec,
    scale,
    params=None,
):
    """
    Modo 'flanks': 2 trazas batch, azul=lw, rojo=rw.
    """
    params = params or {}

    color_lw_fill = "rgba(59,130,246,0.75)"
    color_lw_line = "rgba(29,78,216,0.90)"
    color_rw_fill = "rgba(239,68,68,0.75)"
    color_rw_line = "rgba(185,28,28,0.90)"

    lw_xs, lw_ys = [], []
    rw_xs, rw_ys = [], []

    for _, row in agent_step.iterrows():
        x = float(row.get("pos_x", 0))
        y = float(row.get("pos_y", 0))
        lw = float(row.get("lw", 5))
        rw = float(row.get("rw", 5))
        l2 = float(row.get("lambda2", params.get("lambda2_mean", 1.8)))

        if _SHAPELY:
            for side, bx, by in (
                ("left", lw_xs, lw_ys),
                ("right", rw_xs, rw_ys),
            ):
                coords = _flank_world_coords(
                    x=x,
                    y=y,
                    lw=lw,
                    rw=rw,
                    lambda1=lambda1,
                    lambda2=l2,
                    alpha=alpha,
                    delta=delta,
                    wind_vec=wind_vec,
                    side=side,
                    scale=scale,
                )
                if coords:
                    bx.extend(coords[0] + [None])
                    by.extend(coords[1] + [None])
        else:
            lw_xs.append(x - lw * scale * 0.3)
            lw_ys.append(y)
            rw_xs.append(x + rw * scale * 0.3)
            rw_ys.append(y)

    if _SHAPELY:
        for xs, ys, fill, line, name in (
            (lw_xs, lw_ys, color_lw_fill, color_lw_line, "Flanco izq (lw)"),
            (rw_xs, rw_ys, color_rw_fill, color_rw_line, "Flanco der (rw)"),
        ):
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    fill="toself",
                    fillcolor=fill,
                    line=dict(color=line, width=0.6),
                    name=name,
                    showlegend=True,
                    hoverinfo="skip",
                ))
    else:
        for xs, ys, fill, name in (
            (lw_xs, lw_ys, color_lw_fill, "Flanco izq (lw)"),
            (rw_xs, rw_ys, color_rw_fill, "Flanco der (rw)"),
        ):
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(size=8, color=fill),
                    name=name,
                    showlegend=True,
                    hoverinfo="skip",
                ))


def _render_colorby_mode(
    fig,
    agent_step,
    color_by,
    lambda1,
    alpha,
    delta,
    wind_vec,
    scale,
    params=None,
):
    """
    Modo atributo: grupos por color × 2 flancos.
    """
    params = params or {}

    colors = _assign_agent_colors(agent_step, color_by)

    buffers: dict[str, dict] = {}
    color_order = []

    for idx, row in agent_step.iterrows():
        color = colors.loc[idx]

        if color not in buffers:
            buffers[color] = {
                "lw_xs": [],
                "lw_ys": [],
                "rw_xs": [],
                "rw_ys": [],
            }
            color_order.append(color)

        x = float(row.get("pos_x", 0))
        y = float(row.get("pos_y", 0))
        lw = float(row.get("lw", 5))
        rw = float(row.get("rw", 5))
        l2 = float(row.get("lambda2", params.get("lambda2_mean", 1.8)))

        buf = buffers[color]

        if _SHAPELY:
            for side, bx_k, by_k in (
                ("left", "lw_xs", "lw_ys"),
                ("right", "rw_xs", "rw_ys"),
            ):
                coords = _flank_world_coords(
                    x=x,
                    y=y,
                    lw=lw,
                    rw=rw,
                    lambda1=lambda1,
                    lambda2=l2,
                    alpha=alpha,
                    delta=delta,
                    wind_vec=wind_vec,
                    side=side,
                    scale=scale,
                )
                if coords:
                    buf[bx_k].extend(coords[0] + [None])
                    buf[by_k].extend(coords[1] + [None])
        else:
            buf["lw_xs"].append(x - lw * scale * 0.3)
            buf["lw_ys"].append(y)
            buf["rw_xs"].append(x + rw * scale * 0.3)
            buf["rw_ys"].append(y)

    for color in color_order:
        buf = buffers[color]
        fill = _add_alpha_to_color(color, 0.72)
        line = _add_alpha_to_color(color, 0.95)
        label = _color_label(color, color_by)

        if _SHAPELY:
            for side_name, xs_k, ys_k, show_leg in (
                ("lw", "lw_xs", "lw_ys", True),
                ("rw", "rw_xs", "rw_ys", False),
            ):
                xs = buf[xs_k]
                ys = buf[ys_k]
                if xs:
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        fill="toself",
                        fillcolor=fill,
                        line=dict(color=line, width=0.6),
                        name=label,
                        showlegend=show_leg,
                        legendgroup=str(color),
                        hoverinfo="skip",
                    ))
        else:
            for xs, ys, show_leg in (
                (buf["lw_xs"], buf["lw_ys"], True),
                (buf["rw_xs"], buf["rw_ys"], False),
            ):
                if xs:
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        marker=dict(size=8, color=fill),
                        name=label,
                        showlegend=show_leg,
                        legendgroup=str(color),
                        hoverinfo="skip",
                    ))


def _color_label(color: str, color_by: str) -> str:
    """Etiqueta de leyenda para un color en modo color_by."""
    if color_by == "morphotype":
        inv = {v: k for k, v in MORPHOTYPE_COLORS.items()}
        return inv.get(color, color)

    return color_by


def _add_wind_arrow(
    fig: go.Figure,
    wind_vec: tuple[float, float],
    x_range: list[float],
    y_range: list[float],
) -> None:
    """Agrega flecha de viento dentro del rango visible."""
    wx, wy = wind_vec

    x_span = max(x_range[1] - x_range[0], 1.0)
    y_span = max(y_range[1] - y_range[0], 1.0)

    arrow_x0 = x_range[0] + x_span * 0.08
    arrow_y0 = y_range[0] + y_span * 0.10
    arrow_len = min(x_span, y_span) * 0.12

    fig.add_annotation(
        x=arrow_x0 + wx * arrow_len,
        y=arrow_y0 + wy * arrow_len,
        ax=arrow_x0,
        ay=arrow_y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor=C["accent"],
    )

    fig.add_annotation(
        x=arrow_x0 + wx * arrow_len * 1.55,
        y=arrow_y0 + wy * arrow_len * 1.55,
        xref="x",
        yref="y",
        text="viento",
        showarrow=False,
        font=dict(size=9, color=C["accent"]),
    )


# ── 6. Series de tiempo ───────────────────────────────────────────────────────

def make_timeseries_figure(
    model_df: pd.DataFrame,
    step_marker: int = None,
) -> go.Figure:
    """
    Figura Plotly con N_dunes, calveos acumulados y asimetría media.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "N dunas activas",
            "Calveos acumulados",
            "Asimetría media",
        ],
    )

    if model_df.empty:
        fig.add_annotation(
            text="Sin datos",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=C["muted"]),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        return fig

    series = [
        ("N_dunes",        C["accent"], 1),
        ("n_dunes",        C["accent"], 1),
        ("calving_count",  C["warn"],   2),
        ("mean_asymmetry", "#38A169",   3),
        ("mean_asymmetry_final", "#38A169", 3),
    ]

    used_rows = set()

    for col, color, row in series:
        if row in used_rows:
            continue
        if col not in model_df.columns:
            continue

        fig.add_trace(
            go.Scatter(
                x=model_df.index,
                y=model_df[col],
                line=dict(color=color, width=1.6),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        used_rows.add(row)

        if step_marker is not None:
            fig.add_vline(
                x=step_marker,
                line=dict(color=C["border"], width=1.5, dash="dot"),
                row=row,
                col=1,
            )

    fig.update_layout(**PLOTLY_LAYOUT, height=320)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=C["border"],
        gridwidth=0.5,
        title_text="Paso",
        row=3,
        col=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=C["border"],
        gridwidth=0.5,
    )

    return fig


# ── 7. Histograma de distribución de anchos ───────────────────────────────────

def make_histogram_figure(
    agent_step: pd.DataFrame,
    color_by: str = "morphotype",
) -> go.Figure:
    """Histograma de distribución de anchos."""
    fig = go.Figure()

    widths = _get_width_series(agent_step)

    if agent_step.empty or widths.empty:
        fig.add_annotation(
            text="Sin datos",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=C["muted"]),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=220)
        return fig

    bin_max = widths.max() * 1.1 if not widths.empty else 50
    bin_max = max(float(bin_max), 1.0)
    bin_edges = np.linspace(0, bin_max, 22)

    if color_by in ("morphotype", "flanks") and "morphotype" in agent_step.columns:
        for morph in ["barchan", "transverse", "asymmetric", "pre_calving"]:
            mask = agent_step["morphotype"] == morph
            if not mask.any():
                continue

            counts, _ = np.histogram(widths[mask], bins=bin_edges)

            fig.add_trace(go.Bar(
                x=bin_edges[:-1],
                y=counts,
                width=np.diff(bin_edges),
                name=morph,
                marker=dict(
                    color=MORPHOTYPE_COLORS.get(morph, MORPHOTYPE_COLORS["ghost"]),
                    line=dict(width=0),
                ),
                opacity=0.82,
            ))

        fig.update_layout(barmode="stack")

    else:
        counts, _ = np.histogram(widths, bins=bin_edges)

        fig.add_trace(go.Bar(
            x=bin_edges[:-1],
            y=counts,
            width=np.diff(bin_edges),
            marker=dict(color=C["accent"], line=dict(width=0)),
            showlegend=False,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=220,
        xaxis_title="W_l + W_r (m)",
        yaxis_title="N dunas",
        legend=dict(font=dict(size=10)),
        bargap=0.05,
    )

    return fig


# ── 8. Heatmap del espacio de parámetros ──────────────────────────────────────

def make_heatmap_figure(
    summary: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
) -> go.Figure:
    """Heatmap de métrica promedio por combinación de parámetros."""
    if summary.empty or x_param not in summary.columns or y_param not in summary.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Sin datos — ejecuta generate_demo_data.py",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=C["muted"]),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        return fig

    if x_param == y_param:
        fig = go.Figure()
        fig.add_annotation(
            text="Selecciona ejes X/Y diferentes",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=C["muted"]),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        return fig

    if metric not in summary.columns:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Métrica no disponible: {metric}",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=C["muted"]),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        return fig

    if metric in [x_param, y_param]:
        fig = go.Figure()
        fig.add_annotation(
            text="La métrica debe ser diferente de los ejes",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=C["muted"]),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        return fig

    heat_df = (
        summary
        .groupby([y_param, x_param], as_index=False)[metric]
        .mean()
    )

    pivot = heat_df.pivot(
        index=y_param,
        columns=x_param,
        values=metric,
    )

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlBu_r",
        aspect="auto",
        labels={
            "color": METRIC_LABELS.get(metric, metric),
            "x": PARAM_LABELS.get(x_param, x_param),
            "y": PARAM_LABELS.get(y_param, y_param),
        },
        title=f"<b>{METRIC_LABELS.get(metric, metric)}</b>",
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        coloraxis_colorbar=dict(thickness=10, len=0.75),
    )

    return fig


# ── 9. Coordenadas paralelas ──────────────────────────────────────────────────

def make_parallel_figure(
    summary: pd.DataFrame,
    color_metric: str = "calving_rate",
) -> go.Figure:
    """Figura de coordenadas paralelas para exploración de corridas."""
    if summary.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Sin datos",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=C["muted"]),
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        return fig

    def _dim(label: str, col: str):
        if col not in summary.columns:
            return None

        vals = pd.to_numeric(summary[col], errors="coerce")

        if vals.dropna().empty:
            return None

        vmin = float(vals.min())
        vmax = float(vals.max())

        if vmin == vmax:
            vmax = vmin + 1.0

        return dict(
            label=label,
            values=vals.fillna(vmin),
            range=[vmin, vmax],
        )

    dims = [
        _dim("q_sat",        "qsat"),
        _dim("q₀/q_sat",     "q0ratio"),
        _dim("q_shift",      "qshift_ratio"),
        _dim("λ₂ σ",         "lambda2_std"),
        _dim("N final",      "n_dunes_final"),
        _dim("Ancho medio",  "mean_width_final"),
        _dim("Asimetría",    "mean_asymmetry_final"),
        _dim("Calveos/paso", "calving_rate"),
        _dim("P90 ancho",    "p90_width_final"),
    ]
    dims = [d for d in dims if d is not None]

    if not dims:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        return fig

    col = color_metric if color_metric in summary.columns else "calving_rate"

    if col in summary.columns:
        color_vals = pd.to_numeric(summary[col], errors="coerce").fillna(0)
    else:
        color_vals = pd.Series(np.zeros(len(summary)))

    fig = go.Figure(go.Parcoords(
        line=dict(
            color=color_vals,
            colorscale="RdYlBu_r",
            showscale=True,
            colorbar=dict(
                thickness=8,
                len=0.65,
                title=dict(
                    text=METRIC_LABELS.get(col, ""),
                    font=dict(size=9),
                ),
            ),
        ),
        dimensions=dims,
        labelfont=dict(color=C["muted"], size=10),
    ))

    fig.update_layout(**PLOTLY_LAYOUT, height=320)

    return fig