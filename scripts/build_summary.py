"""
build_summary.py
Reconstruye summary.parquet desde run_index.csv.
Útil si el índice se modificó manualmente o se añadieron corridas fuera del grid.

Uso:
    python scripts/build_summary.py
    python scripts/build_summary.py --data-dir /ruta/personalizada/
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.run_storage import RunStorage


def main():
    parser = argparse.ArgumentParser(description="Reconstruir summary.parquet")
    parser.add_argument("--data-dir", default="resultados/")
    args = parser.parse_args()

    df = RunStorage.build_summary(args.data_dir)
    print(f"summary.parquet reconstruido: {len(df)} corridas")
    print(df[["run_id", "wind_regime", "qsat", "lambda2_std", "n_dunes_final"]].head(10))


if __name__ == "__main__":
    main()
