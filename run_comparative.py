"""Ejecuta la comparación entre diccionario externo y adaptativo.

Uso:
    python run_comparative.py
    python run_comparative.py --rebuild-base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from src.adaptativo import run_comparative
from src.pipeline import main as run_base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4, help="hilos BLAS/OpenMP")
    parser.add_argument(
        "--rebuild-base",
        action="store_true",
        help="regenera bloques 1--3 antes de ejecutar la variante adaptativa",
    )
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with threadpool_limits(limits=args.threads):
        if args.rebuild_base:
            run_base(blocks=(1, 2, 3), root=root, threads=args.threads)
        result = run_comparative(
            root,
            n_timing_repeats=max(1, args.timing_repeats),
            verbose=args.verbose,
        )
    print(result["comparison"].to_string(index=False))
    print("Verificación:", json.dumps(result["checks"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
