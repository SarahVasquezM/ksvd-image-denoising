"""Bloque 3 - Evaluación.

Wrapper de conveniencia: la implementación única vive en `src/evaluacion.py`. Este archivo sólo
reexporta las funciones públicas y permite regenerar los productos de esta carpeta:

    python 03_evaluacion/evaluacion.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluacion import run_experiments, reconstruct_image, make_noisy, mse, psnr, metric_sanity_checks  # noqa: E402,F401

if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    from src.pipeline import DEFAULT_THREADS, run_block3

    with threadpool_limits(limits=DEFAULT_THREADS):
        result = run_block3()
    print(result if not isinstance(result, dict) else {k: v for k, v in result.items() if k in ("summary", "checks", "effective")})
