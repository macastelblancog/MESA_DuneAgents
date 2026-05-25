"""
visualizacion/shared/plots.py
Funciones de visualización del campo de dunas.

Diseño de capas
---------------
Capa 1 — Fuente de datos (independiente):
    - Lista de DuneAgent (modelo en vivo)
    - DataFrame de agent_data.parquet (corrida guardada)

Capa 2 — Este módulo: funciones de dibujo que aceptan ambas fuentes.

Capa 3 — Consumidores:
    - animate_field.py (animación standalone)
    - notebooks (00–04)
    - app_trackB.py (snapshot estático en el panel de detalle)
    - app_trackA.py (futuro)

Dependencias
------------
    matplotlib  (siempre requerida)
    shapely     (opcional: si no está, cae a modo scatter sin polígonos)
    numpy, pandas
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors

try:
    from shapely.affinity import rotate as shapely_rotate
    from shapely.affinity import translate as shapely_translate
    _SHAPELY = True
except ImportError:
    _SHAPELY = False

# ── Paleta de colores por morfotipo ───────────────────────────────────────────

MORPHOTYPE_COLORS = {
    'barchan':    '#4A90D9',   # azul   — forma simétrica estable
    'transverse': '#38A169',   # verde  — cresta ancha, viento unimodal
    'asymmetric': '#DD6B20',   # naranja — en transición
    'pre_calving':'#E53E3E',   # rojo   — a punto de fragmentarse
    'ghost':      '#A0AEC0',   # gris   — sin volumen
}

MORPHOTYPE_ALPHA = {
    'barchan': 0.70, 'transverse': 0.65,
    'asymmetric': 0.75, 'pre_calving': 0.85, 'ghost': 0.30,
}

CMAP_LAMBDA2   = plt.cm.RdYlBu_r
CMAP_ASYMMETRY = plt.cm.plasma
CMAP_WIDTH     = plt.cm.viridis


# ── Normalización de fuentes de datos ─────────────────────────────────────────

def agents_to_dicts(source) -> list[dict]:
    """
    Normaliza la fuente de datos a una lista de dicts con campos estándar.

    Acepta:
        - lista de DuneAgent (modelo en vivo)
        - pandas DataFrame con columnas de agent_data.parquet
        - lista de dicts (uso interno)
    """
    if not source:
        return []

    first = next(iter(source)) if hasattr(source, '__iter__') else None

    # Lista de DuneAgent
    if first is not None and hasattr(first, 'pos') and hasattr(first, 'lw'):
        return [
            {
                'x':          a.pos[0],
                'y':          a.pos[1],
                'lw':         a.lw,
                'rw':         a.rw,
                'lambda2':    a.lambda2,
                'morphotype': a.morphotype,
                'asymmetry':  a.asymmetry,
                'width':      a.width,
            }
            for a in source
            if a.lw + a.rw > 0
        ]

    # DataFrame de parquet
    if hasattr(source, 'iterrows'):
        rows = []
        for _, row in source.iterrows():
            rows.append({
                'x':          float(row.get('pos_x', row.get('x', 0))),
                'y':          float(row.get('pos_y', row.get('y', 0))),
                'lw':         float(row['lw']),
                'rw':         float(row['rw']),
                'lambda2':    float(row.get('lambda2', 2.5)),
                'morphotype': str(row.get('morphotype', 'barchan')),
                'asymmetry':  float(row.get('asymmetry', 0.0)),
                'width':      float(row.get('width', row['lw'] + row['rw'])),
            })
        return rows

    return list(source)   # ya son dicts


# ── Construcción de polígonos para matplotlib ─────────────────────────────────

def _make_world_polygons(agent_dict: dict, model_params: dict) -> tuple | None:
    """
    Construye los polígonos del flanco izquierdo y derecho en coordenadas mundo,
    orientados en la dirección real del viento.

    Retorna (left_coords, right_coords) como arrays numpy de shape (N, 2),
    o None si shapely no está disponible.

    Convención:
        - El polígono canónico se crea en el origen con viento en +y.
        - Se rota al ángulo real del viento.
        - Se traslada a la posición del agente.
    """
    if not _SHAPELY:
        return None

    from shapely.geometry import Polygon

    lw  = agent_dict['lw']
    rw  = agent_dict['rw']
    x   = agent_dict['x']
    y   = agent_dict['y']
    l2  = agent_dict['lambda2']

    l1    = model_params.get('lambda1', 1.5)
    alpha = model_params.get('alpha',   0.05)
    delta = model_params.get('delta',   4.6)

    # ── Geometría canónica centrada en el origen (viento en +y) ───────────────
    L_b  = l1 * (lw + rw)
    H_l  = min(alpha * lw + delta / 2.0, lw * (1.0 - 1e-9))
    H_r  = min(alpha * rw + delta / 2.0, rw * (1.0 - 1e-9))
    L_hl = l2 * lw
    L_hr = l2 * rw

    # Centrar el polígono en y: upwind face en -L_body/2
    yc = -L_b / 2.0

    left_pts = [
        (0,            yc),
        (-lw,          yc),
        (-lw,          yc + L_b + L_hl),
        (-lw + H_l,    yc + L_b + L_hl),
        (-lw + H_l,    yc + L_b),
        (0,            yc + L_b),
    ]
    right_pts = [
        (0,            yc),
        (rw,           yc),
        (rw,           yc + L_b + L_hr),
        (rw - H_r,     yc + L_b + L_hr),
        (rw - H_r,     yc + L_b),
        (0,            yc + L_b),
    ]

    left_poly  = Polygon(left_pts)
    right_poly = Polygon(right_pts)

    # ── Rotar al ángulo real del viento ───────────────────────────────────────
    wind_vec = model_params.get('wind_vec', (0.0, 1.0))
    wx, wy   = wind_vec
    theta_wind_deg   = float(np.rad2deg(np.arctan2(wy, wx)))
    rotation_deg     = theta_wind_deg - 90.0   # canónico apunta a +y = 90°

    left_r  = shapely_rotate(left_poly,  rotation_deg, origin=(0, 0))
    right_r = shapely_rotate(right_poly, rotation_deg, origin=(0, 0))

    # ── Trasladar a posición mundo ────────────────────────────────────────────
    left_placed  = shapely_translate(left_r,  x, y)
    right_placed = shapely_translate(right_r, x, y)

    left_coords  = np.array(left_placed.exterior.coords)
    right_coords = np.array(right_placed.exterior.coords)

    return left_coords, right_coords


# ── Función principal de dibujo del campo ─────────────────────────────────────

def draw_field(
    ax: plt.Axes,
    agents,
    model_params: dict,
    color_by:   str   = 'morphotype',
    show_wind:  bool  = True,
    show_ids:   bool  = False,
    alpha_fill: float = None,
    title:      str   = None,
) -> None:
    """
    Dibuja el campo de dunas en un Axes de matplotlib.

    Parámetros
    ----------
    ax           : Axes de matplotlib donde dibujar
    agents       : fuente de datos (DuneAgent list, DataFrame, o list of dicts)
    model_params : dict con las claves:
                     simwidth, simlength — dimensiones del dominio (m)
                     wind_vec            — vector unitario de viento (wx, wy)
                     lambda1, alpha, delta — geometría (para polígonos)
    color_by     : 'morphotype' | 'lambda2' | 'asymmetry' | 'width'
    show_wind    : si dibujar la flecha de dirección del viento
    show_ids     : si mostrar el unique_id de cada agente (debug)
    alpha_fill   : sobreescribir transparencia (None = por morfotipo)
    title        : título del axes (None = sin título)

    Comportamiento de degradación
    -----------------------------
    Si shapely no está disponible, dibuja scatter plot con tamaño ∝ width.
    Si agents está vacío, dibuja el dominio vacío con mensaje.
    """
    ax.cla()

    simwidth  = model_params.get('simwidth',  800)
    simlength = model_params.get('simlength', 500)
    wind_vec  = model_params.get('wind_vec',  (0.0, 1.0))

    # Marco del dominio
    ax.set_xlim(0, simwidth)
    ax.set_ylim(0, simlength)
    ax.set_facecolor('#F7F8FA')
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_color('#CBD5E0')
    ax.tick_params(colors='#718096', labelsize=9)
    ax.set_xlabel('x (m)', fontsize=9, color='#718096')
    ax.set_ylabel('y (m)', fontsize=9, color='#718096')

    data = agents_to_dicts(agents)

    if not data:
        ax.text(simwidth / 2, simlength / 2, 'Sin dunas activas',
                ha='center', va='center', color='#A0AEC0', fontsize=11)
        if title:
            ax.set_title(title, fontsize=11, color='#2D3436', pad=6)
        return

    # ── Construir colores ──────────────────────────────────────────────────────
    if color_by == 'morphotype':
        colors = [MORPHOTYPE_COLORS.get(d['morphotype'], '#A0AEC0') for d in data]
        alphas = [alpha_fill or MORPHOTYPE_ALPHA.get(d['morphotype'], 0.65)
                  for d in data]
    elif color_by == 'lambda2':
        norm   = mcolors.Normalize(vmin=1.2, vmax=4.5)
        colors = [CMAP_LAMBDA2(norm(d['lambda2'])) for d in data]
        alphas = [alpha_fill or 0.70] * len(data)
    elif color_by == 'asymmetry':
        norm   = mcolors.Normalize(vmin=0.0, vmax=0.6)
        colors = [CMAP_ASYMMETRY(norm(d['asymmetry'])) for d in data]
        alphas = [alpha_fill or 0.70] * len(data)
    else:  # width
        widths = [d['width'] for d in data]
        norm   = mcolors.Normalize(vmin=0, vmax=max(widths) if widths else 50)
        colors = [CMAP_WIDTH(norm(d['width'])) for d in data]
        alphas = [alpha_fill or 0.70] * len(data)

    # ── Dibujar polígonos (si shapely disponible) ──────────────────────────────
    if _SHAPELY:
        patches = []
        patch_colors = []
        patch_alphas = []

        for agent_d, color, alpha in zip(data, colors, alphas):
            result = _make_world_polygons(agent_d, model_params)
            if result is None:
                continue
            left_coords, right_coords = result

            for coords in [left_coords, right_coords]:
                patch = mpatches.Polygon(coords, closed=True)
                patches.append(patch)
                patch_colors.append(color)
                patch_alphas.append(alpha)

        if patches:
            for patch, color, alpha in zip(patches, patch_colors, patch_alphas):
                patch.set_facecolor(color)
                patch.set_alpha(alpha)
                patch.set_edgecolor('white')
                patch.set_linewidth(0.5)
                ax.add_patch(patch)

    else:
        # Fallback: scatter plot sin polígonos
        xs    = [d['x']     for d in data]
        ys    = [d['y']     for d in data]
        sizes = [max(20, d['width'] * 3) for d in data]
        ax.scatter(xs, ys, s=sizes, c=colors,
                   alpha=0.75, edgecolors='white', linewidths=0.5,
                   zorder=3)
        ax.text(simwidth * 0.98, simlength * 0.02,
                '⚠ shapely no disponible\nmostrando scatter',
                ha='right', va='bottom', fontsize=7, color='#A0AEC0')

    # ── Etiquetas de ID (modo debug) ──────────────────────────────────────────
    if show_ids:
        for d in data:
            ax.text(d['x'], d['y'], str(d.get('id', '')),
                    fontsize=6, ha='center', va='center',
                    color='white', fontweight='bold')

    # ── Flecha de viento ──────────────────────────────────────────────────────
    if show_wind:
        wx, wy = wind_vec
        ax_x = simwidth  * 0.06
        ax_y = simlength * 0.06
        arrow_len = min(simwidth, simlength) * 0.08
        ax.annotate('',
            xy=(ax_x + wx * arrow_len, ax_y + wy * arrow_len),
            xytext=(ax_x, ax_y),
            arrowprops=dict(arrowstyle='->', color='#4A90D9',
                            lw=1.8, mutation_scale=14),
        )
        ax.text(ax_x + wx * arrow_len * 1.35,
                ax_y + wy * arrow_len * 1.35,
                'viento', fontsize=8, color='#4A90D9',
                ha='center', va='center')

    # ── Leyenda de morfotipos ─────────────────────────────────────────────────
    if color_by == 'morphotype':
        present = {d['morphotype'] for d in data}
        handles = [
            mpatches.Patch(color=MORPHOTYPE_COLORS[m], label=m,
                           alpha=0.75)
            for m in ['barchan', 'transverse', 'asymmetric', 'pre_calving']
            if m in present
        ]
        if handles:
            ax.legend(handles=handles, loc='upper right',
                      fontsize=8, framealpha=0.8,
                      edgecolor='#CBD5E0')

    # ── Contador de dunas ─────────────────────────────────────────────────────
    ax.text(0.01, 0.99, f'N = {len(data)}',
            transform=ax.transAxes,
            fontsize=9, color='#2D3436',
            va='top', ha='left')

    if title:
        ax.set_title(title, fontsize=11, color='#2D3436', pad=6)


# ── Series de tiempo ──────────────────────────────────────────────────────────

def draw_timeseries(axes, model_df, step_marker: int = None) -> None:
    """
    Dibuja series de tiempo del DataCollector en una lista de Axes.

    Parámetros
    ----------
    axes         : lista de 3 Axes (N_dunes, calveos, asimetría)
    model_df     : DataFrame de model_data.parquet
    step_marker  : si no None, dibuja línea vertical en ese paso
    """
    BLUE   = '#4A90D9'
    ORANGE = '#DD6B20'
    GREEN  = '#38A169'

    config = [
        ('N_dunes',        'N dunas activas',     BLUE),
        ('calving_count',  'Calveos acumulados',   ORANGE),
        ('mean_asymmetry', 'Asimetría media',      GREEN),
    ]

    for ax, (col, label, color) in zip(axes, config):
        ax.cla()
        if col in model_df.columns:
            ax.plot(model_df.index, model_df[col],
                    color=color, linewidth=1.4, alpha=0.9)
            if step_marker is not None:
                ax.axvline(step_marker, color='#CBD5E0',
                           linewidth=1, linestyle='--')
        ax.set_ylabel(label, fontsize=9, color='#718096')
        ax.tick_params(colors='#718096', labelsize=8)
        ax.set_facecolor('#F7F8FA')
        ax.grid(True, color='#E2E5EA', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color('#E2E5EA')


# ── Histograma de distribución de tamaños ─────────────────────────────────────

def draw_histogram(ax, agent_data, color_by: str = 'morphotype') -> None:
    """
    Dibuja la distribución de anchos de la población de dunas.

    Parámetros
    ----------
    ax         : Axes de matplotlib
    agent_data : fuente de datos (DuneAgent list, DataFrame, list of dicts)
    color_by   : 'morphotype' (apilado por tipo) | 'single' (color único)
    """
    ax.cla()
    data = agents_to_dicts(agent_data)

    if not data:
        ax.text(0.5, 0.5, 'Sin datos', transform=ax.transAxes,
                ha='center', va='center', color='#A0AEC0')
        return

    widths = [d['width'] for d in data]

    if color_by == 'morphotype':
        by_type = {}
        for d in data:
            by_type.setdefault(d['morphotype'], []).append(d['width'])

        order  = ['barchan', 'transverse', 'asymmetric', 'pre_calving']
        bins   = np.linspace(0, max(widths) * 1.1, 20)
        bottom = np.zeros(len(bins) - 1)

        for morph in order:
            if morph not in by_type:
                continue
            counts, _ = np.histogram(by_type[morph], bins=bins)
            ax.bar(bins[:-1], counts, width=np.diff(bins),
                   bottom=bottom, align='edge',
                   color=MORPHOTYPE_COLORS[morph],
                   alpha=0.80, label=morph, edgecolor='white',
                   linewidth=0.3)
            bottom += counts

        ax.legend(fontsize=8, framealpha=0.8, edgecolor='#CBD5E0')
    else:
        ax.hist(widths, bins=20, color='#4A90D9',
                alpha=0.75, edgecolor='white', linewidth=0.3)

    ax.set_xlabel('W_l + W_r (m)', fontsize=9, color='#718096')
    ax.set_ylabel('N dunas',       fontsize=9, color='#718096')
    ax.tick_params(colors='#718096', labelsize=8)
    ax.set_facecolor('#F7F8FA')
    ax.grid(True, axis='y', color='#E2E5EA', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color('#E2E5EA')


# ── Snapshot estático (para Track B) ──────────────────────────────────────────

def render_snapshot(agents, model_params: dict,
                    color_by: str = 'morphotype',
                    figsize: tuple = (9, 6)) -> plt.Figure:
    """
    Genera una Figure con el campo de dunas y el histograma lateral.
    Útil para save_snapshot() en RunStorage y para los notebooks.

    Retorna la Figure (no la guarda; el llamador decide qué hacer con ella).
    """
    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.25)
    ax_field = fig.add_subplot(gs[0])
    ax_hist  = fig.add_subplot(gs[1])

    draw_field(ax_field, agents, model_params,
               color_by=color_by, show_wind=True)
    draw_histogram(ax_hist, agents, color_by=color_by)

    fig.patch.set_facecolor('#FFFFFF')
    return fig