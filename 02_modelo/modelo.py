"""Bloque 2 - OMP y K-SVD.

Wrapper de conveniencia: la implementación única vive en `src/modelo.py`. Este archivo sólo
reexporta las funciones públicas y permite regenerar los productos de esta carpeta:

    python 02_modelo/modelo.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modelo import sparse_code, fit_ksvd, init_dictionary, ksvd_dictionary_update, rank1_atom_update, validate_model, train_from_files  # noqa: E402,F401

if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    from src.pipeline import DEFAULT_THREADS, run_block2

    with threadpool_limits(limits=DEFAULT_THREADS):
        result = run_block2()
    print(result if not isinstance(result, dict) else {k: v for k, v in result.items() if k in ("summary", "checks", "effective")})
