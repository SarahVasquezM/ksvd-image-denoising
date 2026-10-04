"""Bloque 2 - Codificación dispersa (OMP) y aprendizaje de diccionario K-SVD.

* La codificación dispersa usa ``sklearn.linear_model.orthogonal_mp`` (no se reimplementa OMP).
* La actualización de cada átomo usa ``numpy.linalg.svd`` sobre el residual RESTRINGIDO a las
  señales que usan el átomo (no se reimplementa la SVD).
* El ciclo K-SVD (alternar codificación / actualización átomo por átomo) es propio.

Convenciones: X (n, N) señales en columnas, D (n, K) diccionario con columnas de norma 1,
A (K, N) coeficientes.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import orthogonal_mp

ZERO_SIGNAL_TOL = 1e-8   # señales con norma <= tol se codifican con ceros
COEF_TOL = 1e-10         # umbral para contar un coeficiente como activo
NORM_TOL = 1e-6          # tolerancia para aceptar columnas de norma 1

_PREMATURE_MSG = "Orthogonal matching pursuit ended prematurely"


class TrainingTooSlow(RuntimeError):
    """Se lanza cuando la proyección de tiempo supera el presupuesto (ver contingencia)."""

    def __init__(self, projected_s: float, budget_s: float, first_iter_s: float):
        super().__init__(f"Tiempo proyectado {projected_s:.1f}s > presupuesto {budget_s:.1f}s "
                         f"(primera iteración: {first_iter_s:.2f}s)")
        self.projected_s = projected_s
        self.budget_s = budget_s
        self.first_iter_s = first_iter_s


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def normalize_columns(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    norms = np.linalg.norm(D, axis=0)
    if np.any(norms <= ZERO_SIGNAL_TOL):
        raise ValueError("No se puede normalizar una columna (casi) nula")
    return D / norms[None, :]


def count_active(A: np.ndarray, tol: float = COEF_TOL) -> np.ndarray:
    """Número de coeficientes con |a| > tol en cada columna."""
    return np.sum(np.abs(A) > tol, axis=0)


# ---------------------------------------------------------------------------
# Codificación dispersa
# ---------------------------------------------------------------------------
def sparse_code(D: np.ndarray, Z: np.ndarray, max_nonzero: int) -> np.ndarray:
    """Codifica cada columna de ``Z`` con a lo más ``max_nonzero`` átomos de ``D`` (OMP).

    Parameters
    ----------
    D : (n, K) diccionario con columnas de norma 1.
    Z : (n, N) señales (o (n,) para una sola señal).
    max_nonzero : número MÁXIMO de coeficientes no nulos por señal.

    Returns
    -------
    A : (K, N) siempre bidimensional (también cuando N = 1).

    Notas
    -----
    * Se llama a ``orthogonal_mp(D, Z, n_nonzero_coefs=max_nonzero)``; OMP no tiene intercepto
      (los parches ya están centrados).
    * Señales con norma <= 1e-8 reciben coeficientes cero sin llamar a OMP.
    * Si OMP termina antes por dependencia lineal / residual nulo, scikit-learn emite un
      ``RuntimeWarning`` por señal afectada; aquí se registra y se re-emite UNA advertencia resumida.
      Cualquier otra advertencia se re-emite intacta.
    """
    D = np.asarray(D, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if D.ndim != 2 or Z.ndim != 2 or D.shape[0] != Z.shape[0]:
        raise ValueError(f"Formas incompatibles D{D.shape}, Z{Z.shape}")
    n, K = D.shape
    max_nonzero = int(max_nonzero)
    if not 1 <= max_nonzero <= min(n, K):
        raise ValueError(f"max_nonzero={max_nonzero} fuera de [1, {min(n, K)}]")
    if not (np.all(np.isfinite(D)) and np.all(np.isfinite(Z))):
        raise ValueError("D y Z deben ser finitos")
    if np.max(np.abs(np.linalg.norm(D, axis=0) - 1.0)) > NORM_TOL:
        raise ValueError("Las columnas de D deben tener norma 1")

    N = Z.shape[1]
    A = np.zeros((K, N), dtype=np.float64)
    active = np.flatnonzero(np.linalg.norm(Z, axis=0) > ZERO_SIGNAL_TOL)
    if active.size == 0:
        return A

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coef = orthogonal_mp(D, Z[:, active], n_nonzero_coefs=max_nonzero, precompute=True)
    coef = np.asarray(coef).reshape(K, active.size)
    A[:, active] = coef

    premature = 0
    for w in caught:
        if issubclass(w.category, RuntimeWarning) and _PREMATURE_MSG in str(w.message):
            premature += 1
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    if premature:
        warnings.warn(f"OMP terminó antes de {max_nonzero} átomos en {premature} señal(es) "
                      "(residual nulo o dependencia lineal); max_nonzero es un máximo.",
                      RuntimeWarning, stacklevel=2)
    return A


# ---------------------------------------------------------------------------
# K-SVD
# ---------------------------------------------------------------------------
def init_dictionary(X: np.ndarray, n_atoms: int, seed: int) -> np.ndarray:
    """Elige ``n_atoms`` columnas no constantes de X (sin reemplazo, RNG ``seed``) y las normaliza."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=0)
    candidates = np.flatnonzero(norms > ZERO_SIGNAL_TOL)
    if candidates.size < n_atoms:
        raise ValueError(f"Sólo {candidates.size} columnas no constantes para {n_atoms} átomos")
    rng = np.random.default_rng(seed)
    idx = rng.choice(candidates, size=n_atoms, replace=False)
    return X[:, idx] / norms[idx][None, :]


def rank1_atom_update(E_omega: np.ndarray):
    """Mejor aproximación de rango 1 de ``E_omega`` vía ``numpy.linalg.svd``.

    Devuelve ``(d, a)`` con ``d = U[:,0]`` (norma 1) y ``a = s[0] * Vt[0,:]``.
    """
    U, s, Vt = np.linalg.svd(E_omega, full_matrices=False)
    return U[:, 0].copy(), s[0] * Vt[0, :]


def ksvd_dictionary_update(X: np.ndarray, D: np.ndarray, A: np.ndarray):
    """Una pasada de actualización K-SVD átomo por átomo (Gauss-Seidel: usa D y A ya actualizados).

    Para cada átomo j:
        omega = {i : A[j, i] != 0}
        E = X[:, omega] - D @ A[:, omega] + outer(D[:, j], A[j, omega])
        U, s, Vt = svd(E);  D[:, j] = U[:, 0];  A[j, omega] = s[0] * Vt[0, :]
    Si omega está vacío, el átomo se reinicializa con la columna de mayor norma del residual de
    entrenamiento actual ``X - D @ A`` (normalizada), sin repetir columnas dentro de la pasada.
    Si todos los residuales son prácticamente nulos, se conserva el átomo.

    Returns ``(D, A, info)`` con ``info = {"unused_atoms", "reinitialized_atoms"}``.
    """
    D = np.array(D, dtype=np.float64, copy=True)
    A = np.array(A, dtype=np.float64, copy=True)
    K = D.shape[1]
    unused = 0
    reinit = 0
    taken: set[int] = set()
    for j in range(K):
        omega = np.flatnonzero(A[j, :] != 0)
        if omega.size == 0:
            unused += 1
            R = X - D @ A
            rn = np.linalg.norm(R, axis=0)
            if taken:
                rn[list(taken)] = -1.0
            i = int(np.argmax(rn))
            if rn[i] > ZERO_SIGNAL_TOL:
                D[:, j] = R[:, i] / rn[i]
                taken.add(i)
                reinit += 1
            continue
        E = X[:, omega] - D @ A[:, omega] + np.outer(D[:, j], A[j, omega])
        d, a = rank1_atom_update(E)
        D[:, j] = d
        A[j, omega] = a
    return D, A, {"unused_atoms": unused, "reinitialized_atoms": reinit}


def fit_ksvd(X_train: np.ndarray, n_atoms: int = 128, n_iter: int = 8, train_sparsity: int = 4,
             seed: int = 43, time_budget_s: float | None = None, verbose: bool = False):
    """Entrena un diccionario con K-SVD.

    Returns
    -------
    D : (n, n_atoms) diccionario final, columnas de norma 1.
    A_train : (n_atoms, N) códigos de X_train recalculados con OMP sobre el D FINAL.
    history : lista de dicts con ``iteration``, ``train_mse`` (tras la actualización del
        diccionario de esa iteración), ``elapsed_s`` (acumulado), ``unused_atoms`` y
        ``reinitialized_atoms``.

    Si ``time_budget_s`` no es None, tras la primera iteración se proyecta el tiempo total
    (``n_iter`` x tiempo de la primera iteración) y se lanza ``TrainingTooSlow`` si lo excede.
    """
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("X_train debe ser una matriz 2-D finita")
    D = init_dictionary(X, n_atoms, seed)
    history = []
    t0 = time.perf_counter()
    for it in range(1, n_iter + 1):
        A = sparse_code(D, X, train_sparsity)
        D, A, info = ksvd_dictionary_update(X, D, A)
        mse = float(np.mean((X - D @ A) ** 2))
        elapsed = time.perf_counter() - t0
        history.append({"iteration": it, "train_mse": mse, "elapsed_s": elapsed, **info})
        if verbose:
            print(f"iter {it}: mse={mse:.6e} t={elapsed:.2f}s sin_uso={info['unused_atoms']}")
        if it == 1 and time_budget_s is not None and elapsed * n_iter > time_budget_s:
            raise TrainingTooSlow(elapsed * n_iter, time_budget_s, elapsed)
    A_train = sparse_code(D, X, train_sparsity)
    return D, A_train, history


def validate_model(D: np.ndarray, A_train: np.ndarray, X_train: np.ndarray, max_nonzero: int) -> dict:
    """Validaciones de contrato del modelo. Lanza ``AssertionError`` si alguna falla."""
    n, N = X_train.shape
    K = D.shape[1]
    recon = D @ A_train
    err = float(np.mean((X_train - recon) ** 2))
    checks = {
        "D_shape_ok": D.shape == (n, K),
        "D_unit_norm_max_dev": float(np.max(np.abs(np.linalg.norm(D, axis=0) - 1.0))),
        "D_finite": bool(np.all(np.isfinite(D))),
        "A_shape_ok": A_train.shape == (K, N),
        "A_finite": bool(np.all(np.isfinite(A_train))),
        "max_active_per_column": int(count_active(A_train).max()),
        "recon_shape_ok": recon.shape == X_train.shape,
        "train_mse_final": err,
    }
    assert checks["D_shape_ok"], "Forma de D incorrecta"
    assert checks["D_unit_norm_max_dev"] < NORM_TOL, "Columnas de D sin norma 1"
    assert checks["D_finite"] and checks["A_finite"], "Valores no finitos"
    assert checks["A_shape_ok"], "Forma de A_train incorrecta"
    assert checks["max_active_per_column"] <= max_nonzero, "Demasiados coeficientes activos"
    assert checks["recon_shape_ok"], "Forma de D @ A_train incorrecta"
    assert np.isfinite(err), "Error de entrenamiento no finito"
    return checks


# ---------------------------------------------------------------------------
# Entrenamiento con contingencia y productos del bloque
# ---------------------------------------------------------------------------
CONTINGENCY_LEVELS = [
    {"level": 0, "n_atoms": 128, "n_train_patches": 1500, "n_iter": 8, "label": "configuración deseada"},
    {"level": 1, "n_atoms": 128, "n_train_patches": 500, "n_iter": 5, "label": "contingencia nivel 1"},
    {"level": 2, "n_atoms": 64, "n_train_patches": 500, "n_iter": 5,
     "label": "contingencia nivel 2 (K=64: el diccionario deja de ser sobrecompleto)"},
]


def train_from_files(data_path, config_path, output_dir, time_budget_s: float = 3600.0,
                     verbose: bool = True) -> dict:
    """Entrena con la configuración de ``config.json`` y escribe ``modelo.npz``,
    ``historial.csv`` y ``config_efectiva.json`` en ``output_dir``.

    Primero se intenta la configuración deseada; sólo si la medición de la primera iteración
    proyecta un tiempo mayor que ``time_budget_s`` se baja al siguiente nivel de contingencia.
    """
    import pandas as pd

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    with np.load(data_path) as d:
        X_full = d["X_train"].astype(np.float64)
        train_indices = d["train_indices"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    desired = {"n_atoms": int(cfg["n_atoms"]), "n_train_patches": int(X_full.shape[1]),
               "n_iter": int(cfg["n_iter"])}
    levels = [dict(CONTINGENCY_LEVELS[0], **desired)] + CONTINGENCY_LEVELS[1:]
    attempts = []
    for lvl in levels:
        X = X_full[:, :lvl["n_train_patches"]]
        t0 = time.perf_counter()
        try:
            D, A_train, history = fit_ksvd(X, n_atoms=lvl["n_atoms"], n_iter=lvl["n_iter"],
                                           train_sparsity=int(cfg["train_sparsity"]),
                                           seed=int(cfg["seeds"]["ksvd"]),
                                           time_budget_s=time_budget_s, verbose=verbose)
        except TrainingTooSlow as e:
            attempts.append({**lvl, "status": "descartado", "reason": str(e)})
            continue
        total = time.perf_counter() - t0
        attempts.append({**lvl, "status": "usado", "total_s": total})
        break
    else:
        raise RuntimeError("Ningún nivel de contingencia cabe en el presupuesto de tiempo")

    D_init = init_dictionary(X, lvl["n_atoms"], int(cfg["seeds"]["ksvd"]))
    checks = validate_model(D, A_train, X, int(cfg["train_sparsity"]))

    np.savez(out / "modelo.npz", D=D, D_init=D_init, A_train=A_train)
    hist_df = pd.DataFrame(history)[["iteration", "train_mse", "elapsed_s", "unused_atoms"]]
    hist_df.to_csv(out / "historial.csv", index=False)

    init_mse = float(np.mean((X - D_init @ sparse_code(D_init, X, int(cfg["train_sparsity"]))) ** 2))
    eff = {
        "version_contrato": cfg.get("version_contrato"),
        "contingency_level": lvl["level"],
        "contingency_label": lvl["label"],
        "contingency_applied": lvl["level"] > 0,
        "overcomplete": lvl["n_atoms"] > X.shape[0],
        "attempts": attempts,
        "time_budget_s": time_budget_s,
        "n_atoms": lvl["n_atoms"],
        "atom_dim": int(X.shape[0]),
        "n_train_patches": int(X.shape[1]),
        "n_iter": lvl["n_iter"],
        "train_sparsity": int(cfg["train_sparsity"]),
        "seed": int(cfg["seeds"]["ksvd"]),
        "train_indices_used": "primeras n_train_patches columnas de X_train (orden del muestreo RNG 42)",
        "omp": "sklearn.linear_model.orthogonal_mp(n_nonzero_coefs=train_sparsity, precompute=True)",
        "svd": "numpy.linalg.svd(E_omega, full_matrices=False)",
        "unused_atom_policy": "reinicializar con la columna de mayor norma del residual X - D A "
                              "(normalizada); conservar si todos los residuales son ~0",
        "training_time_s": attempts[-1]["total_s"],
        "threadpools": _threadpool_summary(),
        "train_mse_init_dictionary": init_mse,
        "train_mse_final_recoded": checks["train_mse_final"],
        "total_reinitialized_atoms": int(sum(h["reinitialized_atoms"] for h in history)),
        "checks": checks,
        "first_train_index": int(train_indices[0]),
    }
    with open(out / "config_efectiva.json", "w", encoding="utf-8") as f:
        json.dump(eff, f, indent=2, ensure_ascii=False)
    return {"D": D, "D_init": D_init, "A_train": A_train, "history": history, "effective": eff}


def _threadpool_summary() -> list:
    from threadpoolctl import threadpool_info
    return [{"api": i.get("user_api"), "lib": i.get("internal_api"), "num_threads": i.get("num_threads")}
            for i in threadpool_info()]


def load_model(path) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


if __name__ == "__main__":  # pragma: no cover
    import argparse
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Entrena el diccionario K-SVD")
    ap.add_argument("--data", default=str(root / "01_datos" / "datos.npz"))
    ap.add_argument("--config", default=str(root / "01_datos" / "config.json"))
    ap.add_argument("--output-dir", default=str(root / "02_modelo"))
    ap.add_argument("--time-budget-s", type=float, default=3600.0)
    args = ap.parse_args()
    res = train_from_files(args.data, args.config, args.output_dir, args.time_budget_s)
    print(json.dumps({k: v for k, v in res["effective"].items() if k != "attempts"}, indent=2,
                     ensure_ascii=False))
