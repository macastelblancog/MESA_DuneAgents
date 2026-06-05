"""
visualization/stored_results/layout.py
Pestaña de resultados almacenados — lee parquet, no ejecuta MESA.

Cambios respecto a la versión anterior
---------------------------------------
- Filtros explícitos para TODOS los parámetros del grid:
    qsat (RangeSlider), q0ratio (RangeSlider), qshift_ratio (RangeSlider),
    lambda2_std (RangeSlider, max=0.6), wind_regime (Checklist), seed (Dropdown)
- Contador de corridas que pasa los filtros actuales
- cb_exploration: ahora filtra por los 6 parámetros antes de pasar al heatmap
- cb_populate_selector: coincide los filtros activos al buscar matches del click
"""

from pathlib import Path

import pandas as pd
from dash import dcc, html, Input, Output, State, no_update

from visualization.shared.callbacks import (
    load_summary, load_run, load_run_cached, get_steps, agents_at_step,
    make_field_figure, make_timeseries_figure,
    make_histogram_figure, make_heatmap_figure, make_parallel_figure,
    METRIC_LABELS, PARAM_LABELS, ALL_REGIMES, C,
)

# ── Estilos locales ───────────────────────────────────────────────────────────

CTRL = {
    "fontSize": 10, "color": C["muted"],
    "textTransform": "uppercase", "letterSpacing": "0.05em",
    "marginBottom": 3, "marginTop": 10, "display": "block",
}

CARD = {
    "background": C["card"], "borderRadius": 8,
    "border": f"1px solid {C['border']}",
    "padding": "12px 16px", "marginBottom": 10,
}

DD = {"fontSize": 12, "marginBottom": 2}

BTN_PLAY = {
    "fontSize": 11, "padding": "4px 14px",
    "cursor": "pointer", "borderRadius": 4,
    "border": f"1px solid {C['border']}",
    "background": "white", "color": C["text"],
    "marginRight": 8,
}

BTN_RESET = {
    "fontSize": 10, "padding": "3px 10px",
    "cursor": "pointer", "borderRadius": 4,
    "border": f"1px solid {C['border']}",
    "background": "white", "color": C["muted"],
}

_BADGE = {
    "display": "inline-block",
    "fontSize": 10, "fontWeight": 600,
    "padding": "2px 8px", "borderRadius": 10,
    "background": C["accent"], "color": "white",
    "marginLeft": 6, "verticalAlign": "middle",
}


# ── Helpers para rangos dinámicos desde el summary ────────────────────────────

def _col_range(summary: pd.DataFrame, col: str,
               fallback: tuple) -> tuple[float, float]:
    """Devuelve (min, max) de una columna numérica o el fallback si no existe."""
    if summary.empty or col not in summary.columns:
        return fallback
    vals = pd.to_numeric(summary[col], errors="coerce").dropna()
    if vals.empty:
        return fallback
    return float(vals.min()), float(vals.max())


def _range_slider(id_: str, lo: float, hi: float,
                  step: float, marks: dict) -> dcc.RangeSlider:
    return dcc.RangeSlider(
        id=id_,
        min=lo, max=hi, step=step,
        value=[lo, hi],
        marks=marks,
        tooltip={"placement": "bottom", "always_visible": False},
        allowCross=False,
    )


# ── Layout ────────────────────────────────────────────────────────────────────

def layout(data_dir: Path) -> html.Div:
    summary = load_summary(data_dir)

    # Regímenes presentes en los datos
    regimes = (
        sorted(summary["wind_regime"].unique().tolist())
        if not summary.empty and "wind_regime" in summary.columns
        else ALL_REGIMES
    )

    # Seeds presentes en los datos
    seeds = (
        sorted(summary["seed"].unique().tolist())
        if not summary.empty and "seed" in summary.columns
        else []
    )
    seed_opts = [{"label": f"seed {s}", "value": s} for s in seeds]

    # Rangos dinámicos — se leen del summary para adaptarse a cualquier grid
    qsat_lo,  qsat_hi  = _col_range(summary, "qsat",         (30.0, 100.0))
    q0_lo,    q0_hi    = _col_range(summary, "q0ratio",       (0.10,  0.50))
    qsh_lo,   qsh_hi   = _col_range(summary, "qshift_ratio",  (0.01,  0.20))
    l2_lo,    l2_hi    = _col_range(summary, "lambda2_std",   (0.00,  0.60))
    # Forzar máximo de lambda2_std a 0.6 por restricción física
    l2_hi = min(l2_hi, 0.60)

    n_total = len(summary)

    return html.Div([

        # Estado persistente
        dcc.Store(id="sr-data-dir",   data=str(data_dir)),
        dcc.Store(id="sr-run-id",     data=None),
        dcc.Store(id="sr-steps",      data=[]),
        dcc.Store(id="sr-run-params", data={}),

        html.Div([

            # ── Panel izquierdo (filtros + exploración) ───────────────────────
            html.Div([
                html.Div([

                    # ── Exploración ───────────────────────────────────────────
                    html.Span("Exploración", style={**CTRL, "fontWeight": 700,
                                                    "marginTop": 0}),

                    html.Span("Eje X", style=CTRL),
                    dcc.Dropdown(
                        id="sr-x-param",
                        options=[{"label": v, "value": k}
                                 for k, v in PARAM_LABELS.items()],
                        value="qsat", clearable=False, style=DD,
                    ),

                    html.Span("Eje Y", style=CTRL),
                    dcc.Dropdown(
                        id="sr-y-param",
                        options=[{"label": v, "value": k}
                                 for k, v in PARAM_LABELS.items()],
                        value="q0ratio", clearable=False, style=DD,
                    ),

                    html.Span("Métrica", style=CTRL),
                    dcc.Dropdown(
                        id="sr-metric",
                        options=[{"label": v, "value": k}
                                 for k, v in METRIC_LABELS.items()],
                        value="n_dunes_final", clearable=False, style=DD,
                    ),

                    html.Span("Color coord. paralelas", style=CTRL),
                    dcc.Dropdown(
                        id="sr-color-metric",
                        options=[{"label": v, "value": k}
                                 for k, v in METRIC_LABELS.items()],
                        value="calving_rate", clearable=False, style=DD,
                    ),

                    # ── Separador ─────────────────────────────────────────────
                    html.Div(style={"height": 1, "background": C["border"],
                                    "margin": "12px 0"}),

                    # ── Filtros ───────────────────────────────────────────────
                    html.Div([
                        html.Span("Filtros", style={**CTRL, "fontWeight": 700,
                                                    "marginTop": 0,
                                                    "display": "inline-block"}),
                        html.Span(id="sr-run-count", children=f"{n_total}",
                                  style=_BADGE),
                        html.Span(" corridas", style={
                            "fontSize": 10, "color": C["muted"],
                            "marginLeft": 4,
                        }),
                    ], style={"marginBottom": 4}),

                    # Régimen de viento
                    html.Span("Régimen de viento", style=CTRL),
                    dcc.Checklist(
                        id="sr-regimes",
                        options=[{"label": f" {r}", "value": r}
                                 for r in regimes],
                        value=regimes,
                        labelStyle={"display": "block", "fontSize": 12,
                                    "marginBottom": 3},
                    ),

                    # q_sat
                    html.Span("q_sat (m²/año)", style=CTRL),
                    _range_slider(
                        "sr-qsat-range", qsat_lo, qsat_hi, step=5.0,
                        marks={
                            qsat_lo: str(int(qsat_lo)),
                            (qsat_lo + qsat_hi) / 2: str(int((qsat_lo + qsat_hi) / 2)),
                            qsat_hi: str(int(qsat_hi)),
                        },
                    ),

                    # q₀ / q_sat
                    html.Span("q₀ / q_sat", style=CTRL),
                    _range_slider(
                        "sr-q0ratio-range", q0_lo, q0_hi, step=0.05,
                        marks={
                            q0_lo: f"{q0_lo:.2f}",
                            (q0_lo + q0_hi) / 2: f"{(q0_lo+q0_hi)/2:.2f}",
                            q0_hi: f"{q0_hi:.2f}",
                        },
                    ),

                    # q_shift / q_sat
                    html.Span("q_shift / q_sat", style=CTRL),
                    _range_slider(
                        "sr-qshift-range", qsh_lo, qsh_hi, step=0.01,
                        marks={
                            qsh_lo: f"{qsh_lo:.2f}",
                            (qsh_lo + qsh_hi) / 2: f"{(qsh_lo+qsh_hi)/2:.2f}",
                            qsh_hi: f"{qsh_hi:.2f}",
                        },
                    ),

                    # λ₂ σ — máx físico 0.6
                    html.Span("λ₂ σ (heterogeneidad)", style=CTRL),
                    _range_slider(
                        "sr-l2std", l2_lo, l2_hi, step=0.05,
                        marks={
                            l2_lo: "0",
                            round((l2_lo + l2_hi) / 2, 2): f"{(l2_lo+l2_hi)/2:.2f}",
                            l2_hi: f"{l2_hi:.2f}",
                        },
                    ),

                    # Seed — filtro por réplica
                    html.Span("Seed (réplica)", style=CTRL),
                    dcc.Dropdown(
                        id="sr-seed-filter",
                        options=seed_opts,
                        value=None,
                        multi=True,
                        placeholder="Todas las semillas",
                        clearable=True,
                        style=DD,
                    ),

                    # Tipo de evento (calveo)
                    html.Span("Tipo de evento dominante", style=CTRL),
                    dcc.Checklist(
                        id="sr-event-filter",
                        options=[
                            {"label": " Calveos",         "value": "calving"},
                            {"label": " Fusiones",        "value": "merging"},
                            {"label": " Intercambios",    "value": "exchange"},
                            {"label": " Fragmentaciones", "value": "fragmentation"},
                        ],
                        value=["calving", "merging", "exchange", "fragmentation"],
                        labelStyle={"display": "block", "fontSize": 12,
                                    "marginBottom": 3},
                    ),

                    # ── Separador ─────────────────────────────────────────────
                    html.Div(style={"height": 1, "background": C["border"],
                                    "margin": "12px 0"}),

                    # ── Visual ────────────────────────────────────────────────
                    html.Span("Visual", style={**CTRL, "fontWeight": 700,
                                               "marginTop": 0}),

                    html.Span("Colorear dunas por", style=CTRL),
                    dcc.Dropdown(
                        id="sr-color-by",
                        options=[
                            {"label": "Flancos (izq/der)", "value": "flanks"},
                            {"label": "Morfotipo",         "value": "morphotype"},
                            {"label": "λ₂",               "value": "lambda2"},
                            {"label": "Asimetría",         "value": "asymmetry"},
                            {"label": "Ancho",             "value": "width"},
                        ],
                        value="flanks", clearable=False, style=DD,
                    ),

                    html.Span("Escala visual de polígonos", style=CTRL),
                    dcc.Slider(
                        id="sr-scale", min=1, max=100, step=1, value=1,
                        marks={1: "auto", 25: "25×", 50: "50×", 100: "100×"},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),

                    html.Span("Vista del campo", style=CTRL),
                    dcc.Dropdown(
                        id="sr-field-view",
                        options=[
                            {"label": "Dominio completo", "value": "domain"},
                            {"label": "Campo activo",     "value": "active"},
                            {"label": "Auto-ajustar a dunas", "value": "auto"},
                        ],
                        value="domain", clearable=False, style=DD,
                    ),

                    html.Button(
                        "↺ Recargar", id="sr-reload", n_clicks=0,
                        style={
                            "marginTop": 14, "width": "100%",
                            "fontSize": 11, "padding": "5px 0",
                            "cursor": "pointer", "borderRadius": 5,
                            "border": f"1px solid {C['border']}",
                            "background": "white", "color": C["muted"],
                        },
                    ),

                ], style={"padding": "8px 12px"}),

                # Heatmap
                html.Div([
                    dcc.Graph(id="sr-heatmap",
                              config={"displayModeBar": False},
                              style={"height": 300}),
                ], style={**CARD, "marginBottom": 6}),

                # Coordenadas paralelas
                html.Div([
                    dcc.Graph(id="sr-parallel",
                              config={"displayModeBar": False},
                              style={"height": 280}),
                ], style=CARD),

            ], style={
                "width": 380, "flexShrink": 0,
                "overflowY": "auto",
                "borderRight": f"1px solid {C['border']}",
                "background": C["card"],
            }),

            # ── Panel derecho (corrida seleccionada + visualizaciones) ────────
            html.Div([

                html.Div([
                    html.P(
                        "↑  Haz clic en el heatmap para explorar corridas",
                        id="sr-run-hint",
                        style={"fontSize": 11, "color": C["muted"], "margin": 0},
                    ),
                    html.Div([
                        html.Span("Corrida seleccionada", style=CTRL),
                        dcc.Dropdown(
                            id="sr-run-selector",
                            options=[], value=None, clearable=False,
                            placeholder="Haz clic en el heatmap...",
                            style={"fontSize": 12},
                        ),
                    ], id="sr-selector-container", style={"display": "none"}),
                    html.Div(id="sr-params-table", style={"marginTop": 8}),
                ], style={**CARD, "marginBottom": 10}),

                html.Div([
                    html.Div([
                        html.Span("Paso de simulación",
                                  style={**CTRL, "marginTop": 0, "marginBottom": 0}),
                        html.Div([
                            html.Button("▶ Play", id="sr-play-btn",
                                        n_clicks=0, style=BTN_PLAY),
                            html.Span("Velocidad (pasos/tick):",
                                      style={"fontSize": 10, "color": C["muted"],
                                             "marginRight": 6}),
                            dcc.Input(
                                id="sr-speed", type="number",
                                value=1, min=1, max=500, step=1,
                                style={"width": 52, "fontSize": 11,
                                       "padding": "2px 6px",
                                       "border": f"1px solid {C['border']}",
                                       "borderRadius": 4, "marginRight": 10},
                            ),
                            html.Span("Intervalo (ms):",
                                      style={"fontSize": 10, "color": C["muted"],
                                             "marginRight": 4}),
                            dcc.Dropdown(
                                id="sr-interval-ms",
                                options=[
                                    {"label": "100ms", "value": 100},
                                    {"label": "250ms", "value": 250},
                                    {"label": "400ms", "value": 400},
                                    {"label": "800ms", "value": 800},
                                ],
                                value=400, clearable=False,
                                style={"width": 90, "fontSize": 11, "marginRight": 10},
                            ),
                            html.Span(id="sr-play-status", children="",
                                      style={"fontSize": 10, "color": C["muted"]}),
                        ], style={"display": "flex", "alignItems": "center",
                                  "marginTop": 6, "marginBottom": 8}),
                    ]),
                    dcc.Slider(
                        id="sr-step-slider", min=0, max=1, step=1, value=0,
                        marks={},
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode="drag",
                    ),
                    dcc.Interval(id="sr-interval", interval=400,
                                 disabled=True, n_intervals=0),
                ], id="sr-slider-container", style={**CARD, "display": "none"}),

                html.Div([
                    html.Div([
                        html.Button("⌖ Reset zoom", id="sr-reset-zoom",
                                    n_clicks=0, style=BTN_RESET,
                                    title="Restaura el zoom al campo completo"),
                    ], style={"display": "flex", "justifyContent": "flex-end",
                               "marginBottom": 4}),
                    dcc.Graph(id="sr-field",
                              config={"displayModeBar": False},
                              style={"height": 420}),
                ], style=CARD),

                html.Div([
                    html.Div([
                        dcc.Graph(id="sr-histogram",
                                  config={"displayModeBar": False},
                                  style={"height": 240}),
                    ], style={**CARD, "flex": 1, "marginBottom": 0}),
                    html.Div([
                        dcc.Graph(id="sr-timeseries",
                                  config={"displayModeBar": False},
                                  style={"height": 240}),
                    ], style={**CARD, "flex": 2, "marginBottom": 0,
                               "marginLeft": 10}),
                ], style={"display": "flex"}),

            ], style={
                "flex": 1, "padding": 10,
                "overflowY": "auto", "background": C["bg"],
            }),

        ], style={
            "display": "flex",
            "height": "calc(100vh - 88px)",
            "overflow": "hidden",
        }),
    ])


def _hint(text: str) -> html.P:
    return html.P(text, style={"fontSize": 11, "color": C["muted"], "margin": 0})


# ── Función de filtrado centralizada ──────────────────────────────────────────

def _apply_filters(
    summary: pd.DataFrame,
    regimes: list,
    qsat_range: list,
    q0ratio_range: list,
    qshift_range: list,
    l2std_range: list,
    seeds: list,
) -> pd.DataFrame:
    """
    Aplica todos los filtros del panel al summary.
    Centralizado aquí para que cb_exploration y cb_populate_selector
    usen exactamente la misma lógica.
    """
    if summary.empty:
        return summary

    mask = pd.Series(True, index=summary.index)

    if regimes and "wind_regime" in summary.columns:
        mask &= summary["wind_regime"].isin(regimes)

    if qsat_range and "qsat" in summary.columns:
        lo, hi = qsat_range
        mask &= summary["qsat"].between(lo, hi)

    if q0ratio_range and "q0ratio" in summary.columns:
        lo, hi = q0ratio_range
        mask &= summary["q0ratio"].between(lo, hi)

    if qshift_range and "qshift_ratio" in summary.columns:
        lo, hi = qshift_range
        mask &= summary["qshift_ratio"].between(lo, hi)

    if l2std_range and "lambda2_std" in summary.columns:
        lo, hi = l2std_range
        mask &= summary["lambda2_std"].between(lo, hi)

    if seeds and "seed" in summary.columns:
        mask &= summary["seed"].isin(seeds)

    return summary[mask]


def _apply_event_filter(
    summary: pd.DataFrame,
    event_types: list,
) -> pd.DataFrame:
    """
    Filtra corridas según qué tipo de evento fue dominante.
    Una corrida pasa si al menos uno de sus contadores seleccionados > 0,
    o si no existen las columnas (no filtra en ese caso).

    event_types: subconjunto de ["calving","merging","exchange","fragmentation"]
    """
    if summary.empty or not event_types:
        return summary

    col_map = {
        "calving":       "calving_count",
        "merging":       "merging_count",
        "exchange":      "exchange_count",
        "fragmentation": "fragmentation_count",
    }

    available = [col_map[e] for e in event_types if col_map[e] in summary.columns]
    if not available:
        return summary  # columnas no existen — no filtrar

    mask = pd.Series(False, index=summary.index)
    for col in available:
        mask |= (summary[col].fillna(0) > 0)
    return summary[mask]


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    # ── Exploración: heatmap + parallel coords ────────────────────────────────
    @app.callback(
        Output("sr-heatmap",    "figure"),
        Output("sr-parallel",   "figure"),
        Output("sr-run-count",  "children"),
        Input("sr-x-param",        "value"),
        Input("sr-y-param",        "value"),
        Input("sr-metric",         "value"),
        Input("sr-color-metric",   "value"),
        Input("sr-regimes",        "value"),
        Input("sr-qsat-range",     "value"),
        Input("sr-q0ratio-range",  "value"),
        Input("sr-qshift-range",   "value"),
        Input("sr-l2std",          "value"),
        Input("sr-seed-filter",    "value"),
        Input("sr-event-filter",   "value"),
        Input("sr-reload",         "n_clicks"),
        State("sr-data-dir",       "data"),
    )
    def cb_exploration(x_p, y_p, metric, color_m,
                       regimes, qsat_r, q0_r, qsh_r, l2_r, seeds,
                       event_types, _reload, ddir):
        summary = load_summary(Path(ddir))
        filtered = _apply_filters(
            summary,
            regimes=regimes or ALL_REGIMES,
            qsat_range=qsat_r,
            q0ratio_range=q0_r,
            qshift_range=qsh_r,
            l2std_range=l2_r,
            seeds=seeds or [],
        )
        # Filtro por tipo de evento dominante
        event_types = event_types or ["calving","merging","exchange","fragmentation"]
        filtered = _apply_event_filter(filtered, event_types)
        n = len(filtered)
        return (
            make_heatmap_figure(filtered, x_p, y_p, metric),
            make_parallel_figure(filtered, color_m),
            str(n),
        )

    # ── Click en heatmap → selector de corrida ────────────────────────────────
    @app.callback(
        Output("sr-run-selector",       "options"),
        Output("sr-run-selector",       "value"),
        Output("sr-selector-container", "style"),
        Output("sr-run-hint",           "style"),
        Input("sr-heatmap",             "clickData"),
        State("sr-x-param",             "value"),
        State("sr-y-param",             "value"),
        State("sr-regimes",             "value"),
        State("sr-qsat-range",          "value"),
        State("sr-q0ratio-range",       "value"),
        State("sr-qshift-range",        "value"),
        State("sr-l2std",               "value"),
        State("sr-seed-filter",         "value"),
        State("sr-data-dir",            "data"),
    )
    def cb_populate_selector(click_data, x_p, y_p,
                              regimes, qsat_r, q0_r, qsh_r, l2_r, seeds,
                              ddir):
        HIDDEN_SEL = {"display": "none"}
        SHOWN_SEL  = {}
        HINT_VIS   = {"fontSize": 11, "color": C["muted"], "margin": 0}
        HINT_HID   = {"display": "none"}

        if not click_data:
            return [], None, HIDDEN_SEL, HINT_VIS

        pt    = click_data["points"][0]
        x_val = pt["x"]
        y_val = pt["y"]

        summary = load_summary(Path(ddir))
        if summary.empty:
            return [], None, HIDDEN_SEL, HINT_VIS

        # Aplicar los mismos filtros del panel
        filtered = _apply_filters(
            summary,
            regimes=regimes or ALL_REGIMES,
            qsat_range=qsat_r,
            q0ratio_range=q0_r,
            qshift_range=qsh_r,
            l2std_range=l2_r,
            seeds=seeds or [],
        )

        # Afinar al punto exacto del heatmap (x_p, y_p)
        mask = pd.Series(True, index=filtered.index)
        for col, val in [(x_p, x_val), (y_p, y_val)]:
            try:
                mask &= (filtered[col] - float(val)).abs() < 1e-6
            except (TypeError, ValueError):
                mask &= filtered[col] == val

        matching = filtered[mask]
        if matching.empty:
            return [], None, HIDDEN_SEL, HINT_VIS

        options = []
        for _, row in matching.iterrows():
            run_id = str(row.get("run_id", ""))
            regime = row.get("wind_regime", "?")
            qsat   = row.get("qsat", "?")
            q0     = row.get("q0ratio", "?")
            l2std  = row.get("lambda2_std", "?")
            seed   = row.get("seed", "?")
            label  = (
                f"{run_id}  ·  {regime}  ·  "
                f"qsat={qsat}  q0={q0}  λ₂σ={l2std}  seed={seed}"
            )
            options.append({"label": label, "value": run_id})

        first = options[0]["value"] if options else None
        return options, first, SHOWN_SEL, HINT_HID

    # ── Carga de corrida ──────────────────────────────────────────────────────
    @app.callback(
        Output("sr-run-id",           "data"),
        Output("sr-steps",            "data"),
        Output("sr-run-params",       "data"),
        Output("sr-params-table",     "children"),
        Output("sr-step-slider",      "min"),
        Output("sr-step-slider",      "max"),
        Output("sr-step-slider",      "marks"),
        Output("sr-step-slider",      "value"),
        Output("sr-slider-container", "style"),
        Output("sr-interval",         "disabled"),
        Output("sr-play-btn",         "children"),
        Input("sr-run-selector",      "value"),
        State("sr-data-dir",          "data"),
    )
    def cb_load_run(run_id, ddir):
        HIDDEN = {**CARD, "display": "none"}
        SHOWN  = CARD

        if not run_id:
            return (None, [], {}, "", 0, 1, {}, 0, HIDDEN, True, "▶ Play")

        try:
            run_data = load_run(Path(ddir), run_id)
        except Exception:
            return (
                None, [], {}, "Error cargando corrida",
                0, 1, {}, 0, HIDDEN, True, "▶ Play"
            )

        steps  = get_steps(run_data["agents"])
        params = run_data["params"]

        param_keys = [
            ("wind_regime",  "Régimen"),
            ("qsat",         "q_sat (m²/año)"),
            ("q0ratio",      "q₀ / q_sat"),
            ("qshift_ratio", "q_shift / q_sat"),
            ("lambda2_mean", "λ₂ media"),
            ("lambda2_std",  "λ₂ σ"),
            ("lambda1",      "λ₁"),
            ("alpha",        "α"),
            ("delta",        "Δ (m)"),
            ("c",            "c"),
            ("w0",           "w₀ (m)"),
            ("n_dunes_init", "N init"),
            ("n_steps",      "N pasos"),
            ("seed",         "seed"),
        ]

        rows = []
        for col, label in param_keys:
            if col not in params:
                continue
            val  = params[col]
            disp = f"{val:.3f}" if isinstance(val, float) else str(val)
            rows.append(html.Tr([
                html.Td(label, style={"color": C["muted"], "paddingRight": 12,
                                      "fontSize": 11, "paddingBottom": 2}),
                html.Td(disp,  style={"fontWeight": 500, "fontSize": 11,
                                      "paddingBottom": 2}),
            ]))

        mid = len(rows) // 2
        params_table = html.Div([
            html.Table(html.Tbody(rows[:mid]),
                       style={"borderCollapse": "collapse", "flex": 1}),
            html.Table(html.Tbody(rows[mid:]),
                       style={"borderCollapse": "collapse", "flex": 1,
                              "marginLeft": 20}),
        ], style={"display": "flex"})

        if not steps:
            return (run_id, [], params, params_table,
                    0, 1, {}, 0, HIDDEN, True, "▶ Play")

        n_marks = min(10, len(steps))
        idx = [int(i * (len(steps) - 1) / max(n_marks - 1, 1))
               for i in range(n_marks)]
        marks = {steps[i]: str(steps[i]) for i in idx}

        return (run_id, steps, params, params_table,
                steps[0], steps[-1], marks, steps[0],
                SHOWN, True, "▶ Play")

    # ── Play / Pausa ──────────────────────────────────────────────────────────
    @app.callback(
        Output("sr-interval",    "disabled",  allow_duplicate=True),
        Output("sr-play-btn",    "children",  allow_duplicate=True),
        Output("sr-play-status", "children"),
        Input("sr-play-btn",     "n_clicks"),
        State("sr-interval",     "disabled"),
        State("sr-step-slider",  "value"),
        State("sr-step-slider",  "max"),
        prevent_initial_call=True,
    )
    def cb_toggle_play(n_clicks, is_disabled, current_step, max_step):
        if not n_clicks:
            return True, "▶ Play", ""
        if not is_disabled:
            return True, "▶ Play", ""
        if current_step is not None and max_step is not None \
                and current_step >= max_step:
            return True, "▶ Play", "Fin — mueve el slider para retroceder"
        return False, "⏸ Pausa", "reproduciendo..."

    # ── Control dinámico de intervalo (ms/tick) ──────────────────────────────
    @app.callback(
        Output("sr-interval", "interval"),
        Input("sr-interval-ms", "value"),
        prevent_initial_call=True,
    )
    def cb_set_interval(ms):
        return int(ms or 400)

    @app.callback(
        Output("sr-step-slider", "value",    allow_duplicate=True),
        Output("sr-interval",    "disabled", allow_duplicate=True),
        Output("sr-play-btn",    "children", allow_duplicate=True),
        Output("sr-play-status", "children", allow_duplicate=True),
        Input("sr-interval",     "n_intervals"),
        State("sr-step-slider",  "value"),
        State("sr-step-slider",  "max"),
        State("sr-speed",        "value"),
        prevent_initial_call=True,
    )
    def cb_advance(n_intervals, current_step, max_step, speed):
        if current_step is None or max_step is None:
            return no_update, True, "▶ Play", ""
        if current_step >= max_step:
            return max_step, True, "▶ Play", "Fin"
        next_step = min(current_step + max(1, int(speed or 1)), max_step)
        return next_step, no_update, no_update, no_update

    # ── Visualización del campo ───────────────────────────────────────────────
    @app.callback(
        Output("sr-field",      "figure"),
        Output("sr-histogram",  "figure"),
        Output("sr-timeseries", "figure"),
        Input("sr-step-slider", "value"),
        Input("sr-color-by",    "value"),
        Input("sr-scale",       "value"),
        Input("sr-field-view",  "value"),
        Input("sr-reset-zoom",  "n_clicks"),
        State("sr-run-id",      "data"),
        State("sr-run-params",  "data"),
        State("sr-data-dir",    "data"),
    )
    def cb_step(step, color_by, scale, field_view, n_clicks_reset,
                run_id, params_store, ddir):
        empty_field = make_field_figure(pd.DataFrame(), {})
        empty_hist  = make_histogram_figure(pd.DataFrame())
        empty_ts    = make_timeseries_figure(pd.DataFrame())

        if not run_id:
            return empty_field, empty_hist, empty_ts

        try:
            # load_run_cached evita releer SQLite en cada tick del slider
            run_data   = load_run_cached(Path(ddir), run_id)
            agent_step = agents_at_step(run_data["agents"], step)
            model_df   = run_data["model"]
            params     = run_data["params"] or params_store or {}
        except Exception:
            return empty_field, empty_hist, empty_ts

        uirevision_key = f"{run_id}_{n_clicks_reset or 0}_{field_view or 'domain'}"

        return (
            make_field_figure(agent_step, params, color_by, step=step,
                              scale=float(scale or 1),
                              field_view=field_view or "domain",
                              uirevision_key=uirevision_key),
            make_histogram_figure(agent_step, color_by),
            make_timeseries_figure(model_df, step_marker=step),
        )