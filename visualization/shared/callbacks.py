"""
visualizacion/shared/callbacks.py
Funciones auxiliares compartidas entre stored_results y real_time.

Responsabilidades
-----------------
1. Carga de datos     — summary, run, agentes por paso
2. Figuras Plotly     — campo de dunas, series de tiempo, histograma,
                        heatmap de parámetros, coordenadas paralelas
3. Sin layout Dash    — este módulo no importa dash, solo plotly + pandas

Convención de figuras
---------------------
Toda función make_*_figure() retorna go.Figure listo para dcc.Graph.
Acepta DataFrames o listas de dicts. Nunca importa DuneAgent directamente.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
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

ALL_REGIMES = ["unimodal", "bimodal_acute", "bimodal_obtuse", "multidirectional"]

MORPHOTYPE_COLORS = {
    "barchan":    "#4A90D9",
    "transverse": "#38A169",
    "asymmetric": "#DD6B20",
    "pre_calving":"#E53E3E",
    "ghost":      "#A0AEC0",
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
    #margin=dict(l=50, r=16, t=36, b=36),
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
    Retorna dict con claves: params, model (DataFrame), agents (DataFrame).
    agent_data tiene MultiIndex (Step, AgentID).
    """
    run_dir = data_dir / "runs" / run_id
    params = {}
    if (run_dir / "params.json").exists():
        with open(run_dir / "params.json") as f:
            params = json.load(f)

    model_df = pd.DataFrame()
    if (run_dir / "model_data.parquet").exists():
        model_df = pd.read_parquet(run_dir / "model_data.parquet")

    agent_df = pd.DataFrame()
    if (run_dir / "agent_data.parquet").exists():
        agent_df = pd.read_parquet(run_dir / "agent_data.parquet")

    return {"params": params, "model": model_df, "agents": agent_df}


def get_steps(agent_df: pd.DataFrame) -> list[int]:
    """Retorna lista ordenada de pasos disponibles en agent_data."""
    if agent_df.empty:
        return []
    idx = agent_df.index
    if isinstance(idx, pd.MultiIndex):
        return sorted(idx.get_level_values("Step").unique().tolist())
    return sorted(idx.unique().tolist())


def agents_at_step(agent_df: pd.DataFrame, step: int) -> pd.DataFrame:
    """Extrae el DataFrame de agentes para un paso específico."""
    if agent_df.empty:
        return pd.DataFrame()
    if isinstance(agent_df.index, pd.MultiIndex):
        try:
            return agent_df.xs(step, level="Step")
        except KeyError:
            return pd.DataFrame()
    return agent_df[agent_df.index == step]


def estimate_wind_vec(wind_regime: str) -> tuple[float, float]:
    """Estima vector de viento dominante desde el nombre del régimen."""
    vectors = {
        "unimodal":          (0.0,  1.0),
        "bimodal_acute":     (0.5,  0.866),
        "bimodal_obtuse":    (0.707, 0.707),
        "multidirectional":  (0.0,  1.0),
    }
    return vectors.get(wind_regime, (0.0, 1.0))


# ── 2. Geometría de polígonos para Plotly ────────────────────────────────────

def _flank_world_coords(
    x: float, y: float,
    lw: float, rw: float,
    lambda1: float, lambda2: float,
    alpha: float, delta: float,
    wind_vec: tuple,
    side: str,
    scale: float = 1.0,
) -> tuple[list, list] | None:
    """
    Calcula coordenadas mundo del polígono de un flanco, orientado al viento.
    scale > 1 agranda el polígono visualmente manteniendo su posición real.
    """
    if not _SHAPELY:
        return None

    lw_s = lw * scale
    rw_s = rw * scale
    L_b = lambda1 * (lw_s + rw_s)
    if side == 'left':
        w   = lw_s
        H   = min(alpha * lw_s + delta / 2.0 * scale, lw_s * (1.0 - 1e-9))
        L_h = lambda2 * lw_s
        pts = [
            (0,        -L_b / 2),
            (-w,       -L_b / 2),
            (-w,        L_b / 2 + L_h),
            (-w + H,    L_b / 2 + L_h),
            (-w + H,    L_b / 2),
            (0,         L_b / 2),
        ]
    else:
        w   = rw_s
        H   = min(alpha * rw_s + delta / 2.0 * scale, rw_s * (1.0 - 1e-9))
        L_h = lambda2 * rw_s
        pts = [
            (0,       -L_b / 2),
            (w,       -L_b / 2),
            (w,        L_b / 2 + L_h),
            (w - H,    L_b / 2 + L_h),
            (w - H,    L_b / 2),
            (0,        L_b / 2),
        ]

    poly = Polygon(pts)

    # Rotar al ángulo del viento (canónico apunta en +y = 90°)
    wx, wy = wind_vec
    theta_deg    = float(np.rad2deg(np.arctan2(wy, wx)))
    rotation_deg = theta_deg - 90.0
    poly = shapely_rotate(poly, rotation_deg, origin=(0, 0))
    poly = shapely_translate(poly, x, y)

    coords = np.array(poly.exterior.coords)
    # Cerrar el polígono repitiendo el primer punto
    xs = coords[:, 0].tolist() + [coords[0, 0]]
    ys = coords[:, 1].tolist() + [coords[0, 1]]
    return xs, ys


# ── 3. Figura del campo de dunas ──────────────────────────────────────────────

def make_field_figure(
    agent_step: pd.DataFrame,
    params: dict,
    color_by: str = "morphotype",
    step: int = None,
    scale: float = 1.0,
) -> go.Figure:
    """
    Figura Plotly del campo de dunas en un paso dado.

    Muestra los polígonos reales de cada duna si shapely está disponible;
    scatter plot si no lo está. Hover muestra propiedades del agente.

    Parámetros
    ----------
    agent_step : DataFrame con agentes en un paso (salida de agents_at_step)
    params     : dict del modelo (simwidth, simlength, lambda1, alpha, delta,
                 wind_regime o wind_vec)
    color_by   : 'morphotype' | 'lambda2' | 'asymmetry' | 'width'
    step       : número de paso para el título
    """
    simwidth  = params.get("simwidth",  600)
    simlength = params.get("simlength", 400)
    wind_vec  = params.get("wind_vec") or estimate_wind_vec(
                    params.get("wind_regime", "unimodal"))
    lambda1   = params.get("lambda1",  1.5)
    alpha     = params.get("alpha",    0.05)
    delta     = params.get("delta",    4.6)

    fig = go.Figure()
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(range=[0, simwidth], showgrid=False,
                   zeroline=False, title="x (m)"),
        yaxis=dict(range=[0, simlength], showgrid=False,
                   zeroline=False, title="y (m)", scaleanchor="x"),
        showlegend=True,
        legend=dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=C["border"], borderwidth=1),
        height=420,
        title=dict(
            text=f"Campo de dunas — paso {step}" if step is not None else "Campo de dunas",
            font=dict(size=12), x=0.5,
        ),
    )

    if agent_step.empty:
        fig.add_annotation(text="Sin agentes en este paso",
                           x=simwidth/2, y=simlength/2,
                           showarrow=False, font=dict(color=C["muted"]))
        return fig

    # ── Colormap según variable ───────────────────────────────────────────────
    def get_color(row):
        if color_by == "morphotype":
            return MORPHOTYPE_COLORS.get(row.get("morphotype", "barchan"), "#A0AEC0")
        if color_by == "lambda2":
            norm = (row.get("lambda2", 2.5) - 1.2) / (4.5 - 1.2)
            r = int(255 * min(1, max(0, 2 * norm)))
            b = int(255 * min(1, max(0, 2 * (1 - norm))))
            return f"rgba({r},100,{b},0.75)"
        if color_by == "asymmetry":
            norm = min(1, row.get("asymmetry", 0) / 0.6)
            g = int(200 * (1 - norm))
            return f"rgba(200,{g},60,0.75)"
        # width
        return C["accent"]

    # ── Dibujar: polígonos (shapely) o scatter (fallback) ────────────────────
    added_morphotypes = set()

    for _, row in agent_step.iterrows():
        x   = float(row.get("pos_x", 0))
        y   = float(row.get("pos_y", 0))
        lw  = float(row.get("lw", 5))
        rw  = float(row.get("rw", 5))
        l2  = float(row.get("lambda2", 2.5))
        morph = str(row.get("morphotype", "barchan"))
        asym  = float(row.get("asymmetry", 0))
        w     = lw + rw
        color = get_color(row)
        hover = (f"<b>{morph}</b><br>"
                 f"W={w:.1f}m  lw={lw:.1f} rw={rw:.1f}<br>"
                 f"λ₂={l2:.2f}  asim={asym:.3f}")

        show_legend = (color_by == "morphotype" and morph not in added_morphotypes)
        if show_legend:
            added_morphotypes.add(morph)

        if _SHAPELY:
            for side in ("left", "right"):
                coords = _flank_world_coords(
                    x, y, lw, rw, lambda1, l2, alpha, delta, wind_vec, side,
                    scale=scale,
                )
                if coords is None:
                    continue
                xs, ys = coords
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines",
                    fill="toself",
                    fillcolor=color,
                    line=dict(color="white", width=0.6),
                    showlegend=(show_legend and side == "left"),
                    legendgroup=morph,
                    name=morph,
                    hovertemplate=hover + "<extra></extra>",
                ))
                show_legend = False   # solo una entrada por agente
        else:
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=max(6, w * 1.5), color=color,
                            line=dict(color="white", width=0.5)),
                showlegend=show_legend,
                legendgroup=morph,
                name=morph,
                hovertemplate=hover + "<extra></extra>",
            ))

    # ── Flecha de dirección del viento ────────────────────────────────────────
    wx, wy   = wind_vec
    arrow_x0 = simwidth * 0.05
    arrow_y0 = simlength * 0.05
    arrow_len = min(simwidth, simlength) * 0.10
    fig.add_annotation(
        x=arrow_x0 + wx * arrow_len, y=arrow_y0 + wy * arrow_len,
        ax=arrow_x0, ay=arrow_y0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2,
        arrowwidth=2, arrowcolor=C["accent"],
    )
    fig.add_annotation(
        x=arrow_x0 + wx * arrow_len * 1.6,
        y=arrow_y0 + wy * arrow_len * 1.6,
        xref="x", yref="y",
        text="viento", showarrow=False,
        font=dict(size=9, color=C["accent"]),
    )

    return fig


# ── 4. Series de tiempo ───────────────────────────────────────────────────────

def make_timeseries_figure(
    model_df: pd.DataFrame,
    step_marker: int = None,
) -> go.Figure:
    """
    Figura Plotly con N_dunes, calveos acumulados y asimetría media.
    step_marker dibuja una línea vertical en el paso actual del slider.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["N dunas activas", "Calveos acumulados", "Asimetría media"],
    )

    series = [
        ("N_dunes",        C["accent"], 1),
        ("calving_count",  C["warn"],   2),
        ("mean_asymmetry", "#38A169",   3),
    ]

    for col, color, row in series:
        if col not in model_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=model_df.index,
            y=model_df[col],
            line=dict(color=color, width=1.6),
            showlegend=False,
        ), row=row, col=1)

        if step_marker is not None:
            fig.add_vline(
                x=step_marker,
                line=dict(color=C["border"], width=1.5, dash="dot"),
                row=row, col=1,
            )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        #margin=dict(l=50, r=10, t=30, b=30),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=C["border"], gridwidth=0.5,
        title_text="Paso", row=3, col=1,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=C["border"], gridwidth=0.5,
    )
    return fig


# ── 5. Histograma de distribución de anchos ───────────────────────────────────

def make_histogram_figure(
    agent_step: pd.DataFrame,
    color_by: str = "morphotype",
) -> go.Figure:
    """
    Histograma apilado por morfotipo (o color único si color_by != 'morphotype').
    """
    fig = go.Figure()

    if agent_step.empty or "width" not in agent_step.columns:
        fig.add_annotation(text="Sin datos", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(color=C["muted"]))
        fig.update_layout(**PLOTLY_LAYOUT, height=220)
        return fig

    widths     = agent_step["width"].astype(float)
    bin_max    = widths.max() * 1.1 if not widths.empty else 50
    bin_edges  = np.linspace(0, bin_max, 22)

    if color_by == "morphotype" and "morphotype" in agent_step.columns:
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
                marker=dict(color=MORPHOTYPE_COLORS[morph],
                            line=dict(width=0)),
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
        #margin=dict(l=50, r=10, t=10, b=36),
    )
    return fig


# ── 6. Heatmap del espacio de parámetros ─────────────────────────────────────

def make_heatmap_figure(
    summary: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str,
) -> go.Figure:
    if summary.empty or x_param not in summary or y_param not in summary:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos — ejecuta generate_demo_data.py",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(color=C["muted"]))
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        return fig

    pivot = (summary.groupby([x_param, y_param])[metric]
                    .mean().reset_index()
                    .pivot(index=y_param, columns=x_param, values=metric))

    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlBu_r",
        aspect="auto",
        labels={"color": METRIC_LABELS.get(metric, metric),
                "x": PARAM_LABELS.get(x_param, x_param),
                "y": PARAM_LABELS.get(y_param, y_param)},
        title=f"<b>{METRIC_LABELS.get(metric, metric)}</b>",
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        coloraxis_colorbar=dict(thickness=10, len=0.75),
    )
    return fig


# ── 7. Coordenadas paralelas ──────────────────────────────────────────────────

def make_parallel_figure(
    summary: pd.DataFrame,
    color_metric: str = "calving_rate",
) -> go.Figure:
    if summary.empty:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        return fig

    def _dim(label, col):
        if col not in summary.columns:
            return None
        return dict(label=label, values=summary[col],
                    range=[float(summary[col].min()),
                           float(summary[col].max())])

    dims = [d for d in [
        _dim("q_sat",        "qsat"),
        _dim("q₀/q_sat",     "q0ratio"),
        _dim("q_shift",      "qshift_ratio"),
        _dim("λ₂ σ",         "lambda2_std"),
        _dim("N final",      "n_dunes_final"),
        _dim("Asimetría",    "mean_asymmetry_final"),
        _dim("Calveos/paso", "calving_rate"),
        _dim("P90 ancho",    "p90_width_final"),
    ] if d is not None]

    col = color_metric if color_metric in summary.columns else "calving_rate"
    fig = go.Figure(go.Parcoords(
        line=dict(color=summary[col], colorscale="RdYlBu_r", showscale=True,
                  colorbar=dict(thickness=8, len=0.65,
                                title=dict(text=METRIC_LABELS.get(col, ""),
                                           font=dict(size=9)))),
        dimensions=dims,
        labelfont=dict(color=C["muted"], size=10),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=320,
        #margin=dict(l=70, r=70, t=36, b=36),
    )
    return fig