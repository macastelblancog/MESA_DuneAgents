"""
visualization/real_time/layout.py
Pestaña de simulación en tiempo real — stub.

Pendiente de implementar cuando DuneSwarm.step() esté completamente
funcional (HornFlux + colisiones). El mecanismo es dcc.Interval que
avanza el modelo un paso por tick y actualiza sr-field via dcc.Store.
"""

from dash import dcc, html
from visualization.shared.callbacks import C


def layout() -> html.Div:
    return html.Div([
        html.Div([
            html.P("Simulación en tiempo real",
                   style={"fontWeight": 700, "fontSize": 15,
                          "marginBottom": 6}),
            html.P(
                "Disponible cuando HornFlux y colisiones estén implementados. "
                "Usa generate_demo_data.py + la pestaña de resultados almacenados "
                "para explorar el modelo mientras tanto.",
                style={"color": C["muted"], "fontSize": 13,
                       "maxWidth": 520, "lineHeight": 1.6}),
        ], style={
            "display": "flex", "flexDirection": "column",
            "alignItems": "center", "justifyContent": "center",
            "height": "calc(100vh - 88px)",
        }),
    ])


def register_callbacks(app):
    pass   # sin callbacks hasta implementar