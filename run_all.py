"""Ejecuta el proyecto completo (bloques 1 a 4).  Uso:  python run_all.py [--threads 4]"""
import sys

from src.pipeline import DEFAULT_THREADS, main

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="hilos BLAS (0 = sin límite)")
    a = ap.parse_args()
    main(tuple(a.blocks), threads=a.threads or None)
    sys.exit(0)
