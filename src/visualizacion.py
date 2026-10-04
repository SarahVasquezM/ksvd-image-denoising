"""Figuras del proyecto (fuente única para los cuatro bloques).

Todas las funciones reciben arreglos/DataFrames ya calculados y guardan un PNG; no recalculan
experimentos.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Paleta categórica (orden fijo) y tintas de texto.
C_CLEAN = "#2a78d6"   # condición limpia
C_NOISY = "#eb6834"   # condición ruidosa
C_BASE = "#52514e"    # referencia (ruidosa sin procesar)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 7,
})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _imshow(ax, img, title, vmin=0.0, vmax=1.0, cmap="gray"):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------------------
# Bloque 1
# ---------------------------------------------------------------------------
def plot_images(train, test, path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [2, 1]})
    _imshow(axes[0], train, f"Entrenamiento: camera {train.shape[0]}x{train.shape[1]}")
    _imshow(axes[1], test, f"Prueba: coins {test.shape[0]}x{test.shape[1]}")
    return _save(fig, path)


def patch_grid(columns: np.ndarray, patch_size: int = 8, n_cols: int = 16, pad: int = 1,
               normalize_each: bool = True) -> np.ndarray:
    """Mosaico de columnas (parches o átomos) para visualizar; cada tesela se reescala a [0,1]."""
    n = columns.shape[1]
    n_rows = int(np.ceil(n / n_cols))
    cell = patch_size + pad
    canvas = np.ones((n_rows * cell + pad, n_cols * cell + pad))
    for k in range(n):
        tile = columns[:, k].reshape(patch_size, patch_size)
        if normalize_each:
            lo, hi = tile.min(), tile.max()
            tile = (tile - lo) / (hi - lo) if hi > lo else np.full_like(tile, 0.5)
        r, c = divmod(k, n_cols)
        canvas[pad + r * cell: pad + r * cell + patch_size, pad + c * cell: pad + c * cell + patch_size] = tile
    return canvas


def plot_patch_sample(train, X_train, train_indices, positions, path, n_show=64, patch_size=8):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    _imshow(axes[0], train, "Ubicación de parches muestreados (primeros 64)")
    for idx in train_indices[:n_show]:
        r, c = positions[idx]
        axes[0].add_patch(plt.Rectangle((c - 0.5, r - 0.5), patch_size, patch_size, fill=False,
                                        edgecolor=C_NOISY, linewidth=0.8))
    _imshow(axes[1], patch_grid(X_train[:, :n_show], patch_size, n_cols=8),
            "Parches centrados (reescalados por tesela)")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Bloque 2
# ---------------------------------------------------------------------------
def plot_dictionary(D, path, title, patch_size=8, n_cols=16):
    fig, ax = plt.subplots(figsize=(8, 8 * np.ceil(D.shape[1] / n_cols) / n_cols + 0.4))
    _imshow(ax, patch_grid(D, patch_size, n_cols=n_cols), title)
    return _save(fig, path)


def plot_dictionaries_side_by_side(D_init, D, path, patch_size=8, n_cols=16):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    _imshow(axes[0], patch_grid(D_init, patch_size, n_cols=n_cols), f"D_init ({D_init.shape[1]} átomos)")
    _imshow(axes[1], patch_grid(D, patch_size, n_cols=n_cols), f"D final K-SVD ({D.shape[1]} átomos)")
    return _save(fig, path)


def plot_training_curve(history_df, path, init_mse=None):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(history_df["iteration"], history_df["train_mse"], marker="o", color=C_CLEAN)
    if init_mse is not None:
        ax.axhline(init_mse, color=C_BASE, linestyle="--", linewidth=1.2)
        ax.text(history_df["iteration"].max(), init_mse, "D_init + OMP", color=INK2, ha="right",
                va="bottom", fontsize=9)
    ax.set_xlabel("Iteración K-SVD")
    ax.set_ylabel("MSE de entrenamiento")
    ax.set_title("Error de representación en X_train (T=4)")
    ax.set_xticks(history_df["iteration"])
    return _save(fig, path)


def plot_atom_usage(A_train, path):
    usage = np.sum(np.abs(A_train) > 1e-10, axis=1)
    order = np.argsort(usage)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(np.arange(usage.size), usage[order], color=C_CLEAN, width=0.8)
    ax.set_xlabel("Átomo (ordenado por uso)")
    ax.set_ylabel("Nº de parches que lo usan")
    ax.set_title(f"Uso de átomos en A_train (mín {usage.min()}, máx {usage.max()})")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Bloque 3
# ---------------------------------------------------------------------------
def plot_reconstructions(rec, results_df, sparsities, path):
    test, noisy = rec["test_image"], rec["noisy_image"]
    psnr = {(r.condition, int(r.T0)): r.psnr_db for r in results_df.itertuples()}
    n = len(sparsities) + 1
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.8))
    _imshow(axes[0, 0], test, "Limpia (referencia)")
    _imshow(axes[1, 0], noisy, f"Ruidosa\n{psnr[('ruidosa_sin_procesar', 0)]:.2f} dB")
    for k, T0 in enumerate(sparsities, start=1):
        _imshow(axes[0, k], rec[f"rec_limpia_T{T0}"], f"Rec. limpia T0={T0}\n{psnr[('limpia', T0)]:.2f} dB")
        _imshow(axes[1, k], rec[f"rec_ruidosa_T{T0}"], f"Rec. ruidosa T0={T0}\n{psnr[('ruidosa', T0)]:.2f} dB")
    return _save(fig, path)


def plot_error_maps(rec, sparsities, path):
    test = rec["test_image"]
    n = len(sparsities)
    fig, axes = plt.subplots(2, n, figsize=(2.8 * n + 0.8, 5.8))
    vmax = max(np.abs(rec[f"rec_{c}_T{t}"] - test).max() for c in ("limpia", "ruidosa") for t in sparsities)
    im = None
    for k, T0 in enumerate(sparsities):
        for r, cond in enumerate(("limpia", "ruidosa")):
            im = axes[r, k].imshow(np.abs(rec[f"rec_{cond}_T{T0}"] - test), cmap="magma", vmin=0, vmax=vmax)
            axes[r, k].set_title(f"|error| {cond} T0={T0}", fontsize=9)
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([]); axes[r, k].grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="error absoluto")
    return _save(fig, path)


def _sparsity_lines(ax, df, column, ylabel, baseline=None):
    for cond, color in (("limpia", C_CLEAN), ("ruidosa", C_NOISY)):
        sub = df[df.condition == cond].sort_values("T0")
        ax.plot(sub.T0, sub[column], marker="o", color=color, label=f"entrada {cond}")
    if baseline is not None:
        ax.axhline(baseline, color=C_BASE, linestyle="--", linewidth=1.2, label="ruidosa sin procesar")
    ax.set_xticks(sorted(df[df.T0 > 0].T0.unique()))
    ax.set_xlabel("T0 (máx. átomos por parche)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)


def plot_psnr_vs_t0(df, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    base = float(df.loc[df.condition == "ruidosa_sin_procesar", "psnr_db"].iloc[0])
    _sparsity_lines(ax, df[df.T0 > 0], "psnr_db", "PSNR vs. limpia (dB)", baseline=base)
    ax.set_title("Calidad de reconstrucción según dispersión")
    return _save(fig, path)


def plot_mse_vs_t0(df, path):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    base = float(df.loc[df.condition == "ruidosa_sin_procesar", "mse"].iloc[0])
    _sparsity_lines(ax, df[df.T0 > 0], "mse", "MSE vs. limpia", baseline=base)
    ax.set_title("MSE según dispersión")
    return _save(fig, path)


def plot_activity_time(df, path):
    d = df[df.T0 > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _sparsity_lines(axes[0], d, "mean_active_coefs", "Coef. activos medios por parche")
    axes[0].set_title("Actividad dispersa")
    _sparsity_lines(axes[1], d, "reconstruction_time_s", "Tiempo (s, mediana)")
    axes[1].set_title("Tiempo de reconstrucción")
    axes[1].set_ylim(bottom=0)
    return _save(fig, path)


def plot_summary(D, history_df, df, rec, sparsities, path):
    """Figura-resumen para el reporte final."""
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, 0])
    _imshow(ax, patch_grid(D, int(np.sqrt(D.shape[0])), n_cols=16), "Diccionario K-SVD")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(history_df.iteration, history_df.train_mse, marker="o", color=C_CLEAN)
    ax.set_title("MSE de entrenamiento", fontsize=10); ax.set_xlabel("iteración")
    ax = fig.add_subplot(gs[0, 2:])
    base = float(df.loc[df.condition == "ruidosa_sin_procesar", "psnr_db"].iloc[0])
    _sparsity_lines(ax, df[df.T0 > 0], "psnr_db", "PSNR (dB)", baseline=base)
    ax.set_title("PSNR vs. T0", fontsize=10)
    best = df[df.condition == "ruidosa"].sort_values("psnr_db").iloc[-1]
    T_best = int(best.T0)
    imgs = [(rec["test_image"], "Limpia"),
            (rec["noisy_image"], f"Ruidosa {base:.2f} dB"),
            (rec[f"rec_ruidosa_T{T_best}"], f"Rec. ruidosa T0={T_best} {best.psnr_db:.2f} dB"),
            (rec[f"rec_limpia_T{max(sparsities)}"], f"Rec. limpia T0={max(sparsities)}")]
    for k, (img, t) in enumerate(imgs):
        _imshow(fig.add_subplot(gs[1, k]), img, t)
    fig.tight_layout()
    return _save(fig, path)
