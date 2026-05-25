"""
visualization/stored_results/layout.py
Pestaña de resultados almacenados — lee parquet, no ejecuta MESA.

Dos paneles
-----------
Panel izquierdo — Espacio de parámetros:
    heatmap 2D + coordenadas paralelas + filtros.
    Clic en heatmap → selecciona corrida.

Panel derecho — Corrida seleccionada:
    Slider manual + Play/Pause automático →
    campo de dunas con polígonos reales +
    distribución de anchos + series de tiempo con marcador.
"""

from pathlib import Path

import pandas as pd
from dash import dcc, html, Input, Output, State, no_update

from visualization.shared.callbacks import (
    load_summary, load_run, get_steps, agents_at_step,
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


# ── Layout ────────────────────────────────────────────────────────────────────

def layout(data_dir: Path) -> html.Div:
    summary = load_summary(data_dir)
    regimes = sorted(summary["wind_regime"].unique().tolist()) \
              if not summary.empty and "wind_regime" in summary else ALL_REGIMES

    return html.Div([

        # Estado persistente
        dcc.Store(id="sr-data-dir",   data=str(data_dir)),
        dcc.Store(id="sr-run-id",     data=None),
        dcc.Store(id="sr-steps",      data=[]),
        dcc.Store(id="sr-run-params", data={}),

        html.Div([

            # ── Panel izquierdo ───────────────────────────────────────────────
            html.Div([

                html.Div([
                    html.Span("Exploración", style={**CTRL, "fontWeight": 700,
                                                    "marginTop": 0}),

                    html.Span("Eje X", style=CTRL),
                    dcc.Dropdown(id="sr-x-param",
                        options=[{"label": v, "value": k}
                                 for k, v in PARAM_LABELS.items()],
                        value="qsat", clearable=False, style=DD),

                    html.Span("Eje Y", style=CTRL),
                    dcc.Dropdown(id="sr-y-param",
                        options=[{"label": v, "value": k}
                                 for k, v in PARAM_LABELS.items()],
                        value="q0ratio", clearable=False, style=DD),

                    html.Span("Métrica", style=CTRL),
                    dcc.Dropdown(id="sr-metric",
                        options=[{"label": v, "value": k}
                                 for k, v in METRIC_LABELS.items()],
                        value="n_dunes_final", clearable=False, style=DD),

                    html.Span("Color coord. paralelas", style=CTRL),
                    dcc.Dropdown(id="sr-color-metric",
                        options=[{"label": v, "value": k}
                                 for k, v in METRIC_LABELS.items()],
                        value="calving_rate", clearable=False, style=DD),

                    html.Div(style={"height": 1, "background": C["border"],
                                    "margin": "12px 0"}),

                    html.Span("Filtros", style={**CTRL, "fontWeight": 700}),

                    html.Span("Régimen de viento", style=CTRL),
                    dcc.Checklist(id="sr-regimes",
                        options=[{"label": f" {r}", "value": r}
                                 for r in regimes],
                        value=regimes,
                        labelStyle={"display": "block", "fontSize": 12,
                                    "marginBottom": 3}),

                    html.Span("λ₂ σ (rango)", style=CTRL),
                    dcc.RangeSlider(id="sr-l2std",
                        min=0.0, max=1.0, step=0.1, value=[0.0, 1.0],
                        marks={0: "0", 0.5: "0.5", 1: "1"},
                        tooltip={"placement": "bottom",
                                 "always_visible": False}),

                    html.Span("Colorear agentes por", style=CTRL),
                    dcc.Dropdown(id="sr-color-by",
                        options=[
                            {"label": "Morfotipo", "value": "morphotype"},
                            {"label": "λ₂",        "value": "lambda2"},
                            {"label": "Asimetría", "value": "asymmetry"},
                            {"label": "Ancho",     "value": "width"},
                        ],
                        value="morphotype", clearable=False, style=DD),

                    html.Span("Escala visual de polígonos", style=CTRL),
                    dcc.Slider(id="sr-scale",
                        min=1, max=100, step=1, value=1,
                        marks={1: "1×", 25: "25×", 50: "50×", 100: "100×"},
                        tooltip={"placement": "bottom",
                                 "always_visible": True}),

                    html.Button("↺ Recargar", id="sr-reload", n_clicks=0,
                        style={"marginTop": 14, "width": "100%",
                               "fontSize": 11, "padding": "5px 0",
                               "cursor": "pointer", "borderRadius": 5,
                               "border": f"1px solid {C['border']}",
                               "background": "white", "color": C["muted"]}),
                ], style={"padding": "8px 12px"}),

                html.Div([
                    dcc.Graph(id="sr-heatmap",
                              config={"displayModeBar": False},
                              style={"height": 300}),
                ], style={**CARD, "marginBottom": 6}),

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

            # ── Panel derecho ─────────────────────────────────────────────────
            html.Div([

                # Selector de corrida + parámetros
                html.Div([
                    html.P("↑  Haz clic en el heatmap para explorar corridas",
                           id="sr-run-hint",
                           style={"fontSize": 11, "color": C["muted"], "margin": 0}),

                    # Dropdown con todas las corridas que cumplen el filtro
                    html.Div([
                        html.Span("Corrida seleccionada", style=CTRL),
                        dcc.Dropdown(
                            id="sr-run-selector",
                            options=[], value=None,
                            clearable=False,
                            placeholder="Haz clic en el heatmap...",
                            style={"fontSize": 12},
                        ),
                    ], id="sr-selector-container", style={"display": "none"}),

                    # Tabla de parámetros de la corrida seleccionada
                    html.Div(id="sr-params-table",
                             style={"marginTop": 8}),

                ], style={**CARD, "marginBottom": 10}),

                # Slider + Play/Pause
                html.Div([
                    html.Div([
                        html.Span("Paso de simulación",
                                  style={**CTRL, "marginTop": 0,
                                         "marginBottom": 0}),
                        html.Div([
                            html.Button("▶ Play", id="sr-play-btn",
                                        n_clicks=0, style=BTN_PLAY),
                            html.Span("Velocidad (pasos/tick):",
                                      style={"fontSize": 10, "color": C["muted"],
                                             "marginRight": 6}),
                            dcc.Input(id="sr-speed", type="number",
                                      value=1, min=1, max=50, step=1,
                                      style={"width": 52, "fontSize": 11,
                                             "padding": "2px 6px",
                                             "border": f"1px solid {C['border']}",
                                             "borderRadius": 4,
                                             "marginRight": 10}),
                            html.Span(id="sr-play-status", children="",
                                      style={"fontSize": 10,
                                             "color": C["muted"]}),
                        ], style={"display": "flex", "alignItems": "center",
                                  "marginTop": 6, "marginBottom": 8}),
                    ]),
                    dcc.Slider(
                        id="sr-step-slider",
                        min=0, max=1, step=1, value=0,
                        marks={},
                        tooltip={"placement": "bottom",
                                 "always_visible": True},
                        updatemode="drag",
                    ),
                    dcc.Interval(
                        id="sr-interval",
                        interval=400,
                        disabled=True,
                        n_intervals=0,
                    ),
                ], id="sr-slider-container",
                   style={**CARD, "display": "none"}),

                html.Div([
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
                    ], style={**CARD, "flex": 2,
                               "marginBottom": 0, "marginLeft": 10}),
                ], style={"display": "flex"}),

            ], style={
                "flex": 1, "padding": 10,
                "overflowY": "auto",
                "background": C["bg"],
            }),

        ], style={
            "display": "flex",
            "height": "calc(100vh - 88px)",
            "overflow": "hidden",
        }),
    ])


def _hint(text: str) -> html.P:
    return html.P(text, style={"fontSize": 11, "color": C["muted"],
                                "margin": 0})


# ── Callbacks ─────────────────────────────────────────────────────────────────

def register_callbacks(app):

    # 1. Heatmap + coordenadas paralelas
    @app.callback(
        Output("sr-heatmap",  "figure"),
        Output("sr-parallel", "figure"),
        Input("sr-x-param",      "value"),
        Input("sr-y-param",      "value"),
        Input("sr-metric",       "value"),
        Input("sr-color-metric", "value"),
        Input("sr-regimes",      "value"),
        Input("sr-l2std",        "value"),
        Input("sr-reload",       "n_clicks"),
        State("sr-data-dir",     "data"),
    )
    def cb_exploration(x_p, y_p, metric, color_m, regimes, l2std, _r, ddir):
        summary = load_summary(Path(ddir))

        if not summary.empty and "wind_regime" in summary.columns:
            regimes = regimes or ALL_REGIMES
            lo, hi  = l2std if l2std else [0.0, 1.0]
            mask = summary["wind_regime"].isin(regimes)
            if "lambda2_std" in summary.columns:
                mask &= summary["lambda2_std"].between(lo, hi)
            summary = summary[mask]

        return (
            make_heatmap_figure(summary, x_p, y_p, metric),
            make_parallel_figure(summary, color_m),
        )

    # 2. Clic en heatmap → llenar dropdown con corridas matching
    @app.callback(
        Output("sr-run-selector",      "options"),
        Output("sr-run-selector",      "value"),
        Output("sr-selector-container","style"),
        Output("sr-run-hint",          "style"),
        Input("sr-heatmap",            "clickData"),
        State("sr-x-param",            "value"),
        State("sr-y-param",            "value"),
        State("sr-regimes",            "value"),
        State("sr-data-dir",           "data"),
    )
    def cb_populate_selector(click_data, x_p, y_p, regimes, ddir):
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

        mask = summary["wind_regime"].isin(regimes or ALL_REGIMES)
        for col, val in [(x_p, x_val), (y_p, y_val)]:
            try:
                mask &= (summary[col] - float(val)).abs() < 1e-6
            except (TypeError, ValueError):
                mask &= summary[col] == val

        matching = summary[mask]
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
            label  = (f"{run_id}  ·  {regime}  ·  "
                      f"qsat={qsat}  q0={q0}  λ₂σ={l2std}  seed={seed}")
            options.append({"label": label, "value": run_id})

        first = options[0]["value"] if options else None
        return options, first, SHOWN_SEL, HINT_HID

    # 3. Dropdown cambia → cargar corrida, actualizar stores y slider
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
            return (None, [], {}, "Error cargando corrida",
                    0, 1, {}, 0, HIDDEN, True, "▶ Play")

        steps  = get_steps(run_data["agents"])
        params = run_data["params"]

        # Tabla de parámetros
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
        # Dividir en dos columnas para no ocupar mucho espacio vertical
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
        idx     = [int(i * (len(steps) - 1) / max(n_marks - 1, 1))
                   for i in range(n_marks)]
        marks   = {steps[i]: str(steps[i]) for i in idx}

        return (run_id, steps, params, params_table,
                steps[0], steps[-1], marks, steps[0],
                SHOWN, True, "▶ Play")

    # 3. Play / Pause
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

    # 4. Avanzar un paso en cada tick
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

    # 5. Actualizar visualizationes al cambiar el paso
    @app.callback(
        Output("sr-field",      "figure"),
        Output("sr-histogram",  "figure"),
        Output("sr-timeseries", "figure"),
        Input("sr-step-slider", "value"),
        Input("sr-color-by",    "value"),
        Input("sr-scale",       "value"),
        State("sr-run-id",      "data"),
        State("sr-run-params",  "data"),
        State("sr-data-dir",    "data"),
    )
    def cb_step(step, color_by, scale, run_id, params, ddir):
        empty_field = make_field_figure(pd.DataFrame(), {})
        empty_hist  = make_histogram_figure(pd.DataFrame())
        empty_ts    = make_timeseries_figure(pd.DataFrame())

        if not run_id:
            return empty_field, empty_hist, empty_ts

        try:
            run_data   = load_run(Path(ddir), run_id)
            agent_step = agents_at_step(run_data["agents"], step)
            model_df   = run_data["model"]
        except Exception:
            return empty_field, empty_hist, empty_ts

        return (
            make_field_figure(agent_step, params, color_by,
                              step=step, scale=float(scale or 1)),
            make_histogram_figure(agent_step, color_by),
            make_timeseries_figure(model_df, step_marker=step),
        )