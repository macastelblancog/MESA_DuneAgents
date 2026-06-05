# ABM de dunas eólicas en MESA

Reimplementación y extensión del modelo de Robson & Baas (2023) en MESA 2.x.

## Estructura

```
src/
├── dune_agent.py       ← DuneAgent (física del agente, λ₂ por agente)
├── dune_swarm.py       ← DuneSwarm (orquestación del sistema)
├── flux_physics.py     ← Ecuaciones escalares + geometría Shapely
├── collision_rules.py  ← Detección y resolución de colisiones
├── gamma_threshold.py  ← γ_c (corrige bug B-01 de Robson & Baas)
├── wind_regimes.py     ← Distribuciones de dirección de viento
└── run_storage.py      ← Guardado/carga de corridas en Parquet

scripts/
├── run_grid.py         ← Grid search ~400 combinaciones × 3 réplicas
└── build_summary.py    ← Reconstruir summary.parquet

tests/
└── test_smoke.py       ← Tests básicos (pytest)
```

## Instalación


* Solo el modelo
```bash
pip install -e "."                  
```

* Modelo + Dash
```bash            
pip install -e ".[viz]" 
```

* Modelo + Dash + Jupyter
```bash            
pip install -e ".[viz,notebooks]"
```

* Todo
```bash            
pip install -e ".[all]"
```
## Uso rápido

```python
from src.dune_swarm import DuneSwarm

# Modelo homogéneo (lambda2_std=0 → réplica exacta de Robson & Baas)
model = DuneSwarm(n_dunes_init=20, qsat=100, lambda2_std=0.0, seed=42)
for _ in range(500):
    model.step()

df = model.datacollector.get_model_vars_dataframe()

# Modelo heterogéneo (extensión original)
model2 = DuneSwarm(n_dunes_init=20, qsat=100, lambda2_std=0.5, seed=42)
```


Aplicación

```bash
python visualizacion/app.py
```

```bash
python visualization/app.py --data resultados/
```

## Correcciones sobre el código publicado

| Bug | Archivo original | Descripción | Corrección |
|-----|-----------------|-------------|------------|
| B-01 | GammaStuff.py | lambda1 ausente en gamma_c | Incluido en gamma_threshold.gamma_c() |
| B-02 | ABModel.py | super().__init__ faltante | Corregido en DuneAgent.__init__ |
| B-04 | ABModel.py | lambda2 usado en vez de lambda3 en split() | Corregido en DuneAgent._calve() |

## Extensión original

Cada `DuneAgent` tiene su propio `lambda2 ~ N(μ, σ)` en lugar del valor
global uniforme de Robson & Baas. Esto permite explorar:
- Efecto de la dispersión geométrica en distribuciones de tamaño
- Segregación espacial por λ₂ bajo viento bimodal
- Tiempo de relajación del campo tras cambio de régimen de viento


 python -m pytest tests/ -v > ./tests/testresults.txt
