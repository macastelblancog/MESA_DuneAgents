"""
rebuild_summary.py
Reconstruye resultados/summary.parquet a partir de resultados/runs/*/

Puede ejecutarse como script standalone o llamarse desde run_grid.py
a través de RunStorage.rebuild_summary().

Correcciones
------------
B4  n_dunes_final ahora filtra al último paso del MultiIndex(Step, AgentID)
    antes de contar filas. La versión anterior contaba
    n_pasos × n_agentes en vez de n_agentes_finales.
"""

import sys
from pathlib import Path

# Permite ejecutar como script desde la raíz del proyecto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_storage import RunStorage


DATA_DIR = Path("resultados")


def main():
    storage = RunStorage(DATA_DIR)
    df = storage.rebuild_summary()

    print(f"Summary reconstruido : {DATA_DIR / 'summary.parquet'}")
    print(f"Corridas encontradas : {len(df)}")

    if not df.empty:
        print(f"\nPrimeras 5 filas:")
        cols = ["run_id", "wind_regime", "qsat", "q0ratio",
                "lambda2_std", "n_dunes_final", "calving_rate", "seed"]
        cols_present = [c for c in cols if c in df.columns]
        print(df[cols_present].head().to_string(index=False))


if __name__ == "__main__":
    main()