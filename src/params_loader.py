"""
params_loader.py
Carga, validación y conversión de archivos JSON de parámetros.

Permite configurar DuneSwarm desde un archivo JSON en lugar de pasar
todos los argumentos a mano. Soporta dos presets oficiales:
    params/paper_2023_grl.json  — colisiones individuales (GRL 2023)
    params/paper_2024_esd.json  — enjambres (ESD 2024, estado estable)

Uso básico:
    from src.params_loader import load_params
    p = load_params('params/paper_2024_esd.json')
    swarm = DuneSwarm.from_params(p, seed=42)

Uso desde script:
    python scripts/run_from_json.py --params params/paper_2024_esd.json
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any


# ── Estructura esperada del JSON ──────────────────────────────────────────────

_REQUIRED_SECTIONS = {'geometry', 'flux', 'migration', 'time', 'domain',
                      'injection', 'wind', 'initial_conditions', 'model'}

_DEFAULTS: dict[str, Any] = {
    'geometry': {
        'lambda1': 1.0,
        'lambda2_mean': 1.8,
        'lambda2_std': 0.0,
        'lambda3': 1.0 / 6.0,
        'alpha': 0.05,
        'delta': 4.6,
    },
    'flux': {
        'qsat': 79.0,
        'q0ratio': 0.25,
        'qshift_ratio': 0.10,
        'outflux_mode': 'Hersen',
        'a': 0.45,
        'b': 1.0,
    },
    'migration': {
        'c': 45.0,
        'w0': 16.6,
    },
    'time': {
        'dt': 0.125,
        'n_steps': 100,
    },
    'domain': {
        'simwidth': 600.0,
        'simlength': 600.0,
        'fieldwidth': 200.0,
        'fieldlength': 200.0,
    },
    'injection': {
        'inject': False,
        'inject_mode': 'wmin',
        'rho0': 0.0,
        'w_inject_fixed': None,
    },
    'wind': {
        'regime': 'unimodal',
        'mean_deg': 270.0,
        'std_deg': 3.0,
        'secondary_deg': None,
        'secondary_std_deg': None,
        'secondary_weight': 0.25,
    },
    'initial_conditions': {
        'n_dunes_init': 10,
        'lw_init': [],
        'rw_init': [],
        'x_init': [],
        'y_init': [],
    },
    'model': {
        'collisions': True,
        'seed': None,
    },
}


# ── Funciones públicas ────────────────────────────────────────────────────────

def load_params(path: str | Path) -> dict[str, Any]:
    """Carga un JSON de parámetros, rellena defaults y valida.

    Parámetros
    ----------
    path : ruta al archivo JSON (relativa o absoluta)

    Retorna
    -------
    dict anidado con todos los parámetros, secciones y campos completados.

    Lanza
    -----
    FileNotFoundError si el archivo no existe.
    ValueError si la estructura del JSON es inválida.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo de parámetros no encontrado: {path}\n"
            f"Presets disponibles:\n"
            f"  params/paper_2023_grl.json\n"
            f"  params/paper_2024_esd.json"
        )

    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Rellenar con defaults (no reemplaza valores existentes)
    params = _deep_merge(_DEFAULTS, raw)
    _validate(params, path)
    return params


def save_params(params: dict[str, Any], path: str | Path) -> None:
    """Guarda un dict de parámetros a JSON con formato legible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"Parámetros guardados en: {path}")


def params_to_swarm_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    """Convierte el dict de parámetros a kwargs para DuneSwarm.__init__().

    Retorna
    -------
    dict listo para DuneSwarm(**params_to_swarm_kwargs(p))
    """
    g = params['geometry']
    f = params['flux']
    m = params['migration']
    t = params['time']
    d = params['domain']
    inj = params['injection']
    w = params['wind']
    ic = params['initial_conditions']
    mod = params['model']

    # Calcular w_min teórico
    w_min = (g['delta'] / 2.0) / (1.0 - g['alpha'])

    # Posiciones iniciales: si no se especifican, generar aleatoriamente
    lws = list(ic.get('lw_init', []))
    rws = list(ic.get('rw_init', []))
    xs = list(ic.get('x_init', []))
    ys = list(ic.get('y_init', []))

    return dict(
        # Geometría
        lambda1=g['lambda1'],
        lambda2_mean=g['lambda2_mean'],
        lambda2_std=g['lambda2_std'],
        lambda3=g['lambda3'],
        alpha=g['alpha'],
        delta=g['delta'],
        # Flujo
        qsat=f['qsat'],
        q0ratio=f['q0ratio'],
        qshift_ratio=f['qshift_ratio'],
        outflux_mode=f['outflux_mode'],
        a_duran=f['a'],
        b_duran=f['b'],
        # Migración
        c=m['c'],
        w0=m['w0'],
        # Tiempo
        dt=t['dt'],
        n_steps=t['n_steps'],
        # Dominio
        simwidth=d['simwidth'],
        simlength=d['simlength'],
        fieldwidth=d['fieldwidth'],
        fieldlength=d['fieldlength'],
        # Inyección
        inject=inj['inject'],
        inject_mode=inj.get('inject_mode', 'wmin'),
        rho0=inj.get('rho0', 0.0),
        w_inject_fixed=inj.get('w_inject_fixed'),
        # Viento
        wind_regime=w['regime'],
        wind_mean_deg=w['mean_deg'],
        wind_std_deg=w['std_deg'],
        wind_secondary_deg=w.get('secondary_deg'),
        wind_secondary_std_deg=w.get('secondary_std_deg'),
        wind_secondary_weight=w.get('secondary_weight', 0.25),
        # Modelo
        collisions=mod.get('collisions', True),
        seed=mod.get('seed'),
        # Condiciones iniciales
        lws_init=lws,
        rws_init=rws,
        xs_init=xs,
        ys_init=ys,
        n_dunes_init=ic.get('n_dunes_init', 0),
        # Derivado
        w_min=w_min,
    )


def describe_params(params: dict[str, Any]) -> str:
    """Retorna un resumen legible de los parámetros."""
    meta = params.get('_meta', {})
    g = params['geometry']
    f = params['flux']
    t = params['time']
    d = params['domain']
    lines = [
        f"━━ {meta.get('paper', 'Sin nombre')} ━━",
        f"  DOI        : {meta.get('doi', 'N/A')}",
        f"  λ₁={g['lambda1']}  λ₂={g['lambda2_mean']}  λ₃={g['lambda3']:.4f}  "
        f"α={g['alpha']}  Δ={g['delta']} m",
        f"  qsat={f['qsat']} m²/yr  q₀/qsat={f['q0ratio']}  "
        f"qshift/qsat={f['qshift_ratio']}  mode={f['outflux_mode']}",
        f"  dt={t['dt']} yr  n_steps={t['n_steps']}  "
        f"({t['dt']*t['n_steps']:.1f} años simulados)",
        f"  dominio: {d['simwidth']}m × {d['simlength']}m  "
        f"campo: {d['fieldwidth']}m × {d['fieldlength']}m",
    ]
    if meta.get('note_lambda3'):
        lines.append(f"  NOTA λ₃: {meta['note_lambda3'][:80]}...")
    return '\n'.join(lines)


# ── Utilidades privadas ───────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Fusión profunda: override tiene precedencia, base rellena lo que falta."""
    result = {}
    for k, v_base in base.items():
        if k in override:
            if isinstance(v_base, dict) and isinstance(override[k], dict):
                result[k] = _deep_merge(v_base, override[k])
            else:
                result[k] = override[k]
        else:
            result[k] = v_base
    # Añadir claves presentes solo en override (como '_meta')
    for k in override:
        if k not in result:
            result[k] = override[k]
    return result


def _validate(params: dict, path: Path) -> None:
    """Valida la estructura mínima del dict de parámetros."""
    for section in _REQUIRED_SECTIONS:
        if section not in params:
            raise ValueError(
                f"Sección '{section}' faltante en {path}. "
                f"Requeridas: {_REQUIRED_SECTIONS}"
            )

    g = params['geometry']
    if g['lambda1'] <= 0 or g['lambda2_mean'] <= 0 or g['lambda3'] <= 0:
        raise ValueError("lambda1, lambda2_mean y lambda3 deben ser positivos.")
    if g['alpha'] >= 1.0 or g['alpha'] <= 0.0:
        raise ValueError("alpha debe estar en (0, 1).")
    if g['delta'] <= 0:
        raise ValueError("delta debe ser positivo.")

    f = params['flux']
    if f['outflux_mode'] not in {'Hersen', 'Duran'}:
        raise ValueError(
            f"outflux_mode debe ser 'Hersen' o 'Duran', recibido '{f['outflux_mode']}'"
        )
    if not (0.0 < f['q0ratio'] < 1.0):
        raise ValueError("q0ratio debe estar en (0, 1).")

    inj = params['injection']
    valid_modes = {'wmin', 'weq', 'fixed'}
    if inj.get('inject_mode', 'wmin') not in valid_modes:
        raise ValueError(f"inject_mode debe ser uno de {valid_modes}.")
