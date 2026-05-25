"""
visualizacion/app.py
Aplicación Dash unificada del ABM de dunas barchán.

Pestañas
--------
  stored_results  — exploración del espacio de parámetros y replay de corridas
  real_time       — simulación en vivo (stub, pendiente de implementar)

Uso
---
    cd proyecto_dunas_mesa
    python visualizacion/app.py
    python visualizacion/app.py --data resultados/ --port 8050
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dash
from dash import dcc, html, Input, Output

from visualizacion.stored_results.layout import (
    layout as stored_layout,
    register_callbacks as stored_callbacks,
)
from visualizacion.real_time.layout import (
    layout as realtime_layout,
    register_callbacks as realtime_callbacks,
)
from visualizacion.shared.callbacks import C

# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="ABM Dunas Barchán",
    suppress_callback_exceptions=True,   # necesario con layouts en pestañas
    assets_folder=str(Path(__file__).parent / "assets"),
)

# ── Layout raíz ───────────────────────────────────────────────────────────────

app.layout = html.Div([

    # Header
    html.Div([
        html.Span("ABM Dunas Barchán",
                  style={"fontWeight": 700, "fontSize": 14,
                         "color": C["text"]}),
        html.Span(" · Explorador interactivo",
                  style={"fontSize": 13, "color": C["muted"]}),
    ], style={
        "padding": "9px 18px",
        "background": C["card"],
        "borderBottom": f"1px solid {C['border']}",
        "display": "flex", "alignItems": "center", "gap": 4,
    }),

    # Pestañas
    dcc.Tabs(
        id="main-tabs",
        value="stored",
        style={"borderBottom": f"1px solid {C['border']}"},
        colors={"border": C["border"], "primary": C["accent"],
                "background": C["bg"]},
        children=[
            dcc.Tab(
                label="📦  Resultados almacenados",
                value="stored",
                style={"fontSize": 12, "padding": "8px 18px"},
                selected_style={"fontSize": 12, "padding": "8px 18px",
                                "fontWeight": 600, "borderTop": f"2px solid {C['accent']}"},
            ),
            dcc.Tab(
                label="▶  Simulación en vivo",
                value="realtime",
                style={"fontSize": 12, "padding": "8px 18px"},
                selected_style={"fontSize": 12, "padding": "8px 18px",
                                "fontWeight": 600, "borderTop": f"2px solid {C['accent']}"},
            ),
        ],
    ),

    # Contenido de la pestaña activa
    html.Div(id="tab-content"),

], style={
    "fontFamily": "'Inter', 'Segoe UI', 'Arial', sans-serif",
    "fontSize": 13, "color": C["text"], "background": C["bg"],
})


# ── Callback de navegación entre pestañas ─────────────────────────────────────

def register_tab_callback(data_dir: Path):
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(tab):
        if tab == "stored":
            return stored_layout(data_dir)
        return realtime_layout()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Explorador interactivo ABM de dunas barchán"
    )
    parser.add_argument("--data",  type=str, default="resultados/",
                        help="Directorio de resultados (default: resultados/)")
    parser.add_argument("--port",  type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Registrar callbacks
    register_tab_callback(data_dir)
    stored_callbacks(app)
    realtime_callbacks(app)

    print(f"[app] Datos : {data_dir}")
    print(f"[app] URL   : http://localhost:{args.port}")
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()