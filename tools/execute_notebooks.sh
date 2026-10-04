#!/usr/bin/env bash
# Regenera y ejecuta los cuatro notebooks en orden (cada uno regenera los productos de su bloque).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python "$ROOT/tools/build_notebooks.py"
for nb in 01_datos/01_datos.ipynb 02_modelo/02_modelo.ipynb 03_evaluacion/03_experimentos.ipynb 04_final/04_final.ipynb; do
  echo "== $nb"
  (cd "$ROOT/$(dirname "$nb")" && jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1200 "$(basename "$nb")")
done
