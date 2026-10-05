"""Diccionario K-SVD adaptativo aprendido desde la propia imagen ruidosa.

Esta ruta complementa el experimento original, que aprende un diccionario externo con
``camera`` limpia.  La variante adaptativa sigue la idea de Elad y Aharon (2006): extrae los
parches de la observación corrupta y alterna OMP/K-SVD sobre esos mismos ejemplos antes de
reconstruir la imagen.  Para aislar el efecto del origen de los datos se conservan K, número de
iteraciones, dispersión de entrenamiento, T0 de evaluación, ruido y ensamblado del protocolo
original. La referencia limpia nunca entra al entrenamiento; sólo se usa para calcular métricas.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .evaluacion import make_noisy, run_experiments
from .modelo import ZERO_SIGNAL_TOL, fit_ksvd, init_dictionary, sparse_code, validate_model
from .preprocesamiento import extract_patches


def extract_adaptive_training_patches(
    noisy_image: np.ndarray,
    patch_size: int = 8,
    stride: int = 4,
    min_norm: float = ZERO_SIGNAL_TOL,
) -> dict:
    """Extrae parches centrados no constantes de la observación ruidosa.

    Devuelve la matriz ``X_train`` con señales por columnas, las medias y posiciones
    correspondientes, y los índices dentro de la extracción completa. No hay clipping ni acceso a
    la imagen limpia.
    """
    noisy = np.asarray(noisy_image, dtype=np.float64)
    if noisy.ndim != 2 or not np.all(np.isfinite(noisy)):
        raise ValueError("noisy_image debe ser una matriz 2-D finita")
    Z, means, positions = extract_patches(noisy, patch_size, stride)
    keep = np.flatnonzero(np.linalg.norm(Z, axis=0) > float(min_norm))
    if keep.size == 0:
        raise ValueError("La imagen ruidosa no contiene parches centrados no constantes")
    return {
        "X_train": Z[:, keep],
        "means": means[keep],
        "positions": positions[keep],
        "indices": keep,
        "n_patches_total": int(Z.shape[1]),
    }


def train_adaptive_from_files(data_path, config_path, output_dir, verbose: bool = False) -> dict:
    """Entrena y guarda el diccionario adaptativo usando la observación ruidosa de ``coins``.

    Los productos se escriben en una carpeta nueva para no sobrescribir el modelo externo aceptado:
    ``modelo_adaptativo.npz``, ``historial_adaptativo.csv`` y ``config_adaptativa.json``.
    """
    import pandas as pd

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    with np.load(data_path) as data:
        clean = data["test_image"].astype(np.float64)

    sigma = float(cfg["sigma"])
    noise_seed = int(cfg["seeds"]["noise"])
    model_seed = int(cfg["seeds"].get("ksvd", cfg["seeds"]["model"]))
    noisy, noise = make_noisy(clean, sigma, noise_seed)
    patches = extract_adaptive_training_patches(
        noisy,
        patch_size=int(cfg["patch_size"]),
        stride=int(cfg["stride"]),
        min_norm=float(cfg.get("min_patch_norm", ZERO_SIGNAL_TOL)),
    )
    X = patches["X_train"]
    n_atoms = int(cfg["n_atoms"])
    n_iter = int(cfg["n_iter"])
    train_sparsity = int(cfg["train_sparsity"])
    if X.shape[1] < n_atoms:
        raise ValueError(f"Sólo hay {X.shape[1]} parches válidos para {n_atoms} átomos")

    D_init = init_dictionary(X, n_atoms, model_seed)
    initial_mse = float(np.mean((X - D_init @ sparse_code(D_init, X, train_sparsity)) ** 2))
    started = time.perf_counter()
    D, A_train, history = fit_ksvd(
        X,
        n_atoms=n_atoms,
        n_iter=n_iter,
        train_sparsity=train_sparsity,
        seed=model_seed,
        verbose=verbose,
    )
    training_time_s = time.perf_counter() - started
    checks = validate_model(D, A_train, X, train_sparsity)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "modelo_adaptativo.npz",
        D=D,
        D_init=D_init,
        A_train=A_train,
        training_patch_indices=patches["indices"],
    )
    hist = pd.DataFrame(history)[
        ["iteration", "train_mse", "elapsed_s", "unused_atoms", "reinitialized_atoms"]
    ]
    hist.to_csv(out / "historial_adaptativo.csv", index=False)

    meta = {
        "variant": "diccionario_adaptativo_desde_imagen_ruidosa",
        "paper_basis": "Elad y Aharon (2006), Sec. III-B: Training on the Corrupted Image",
        "training_source": "parches centrados de coins ruidosa",
        "same_noisy_image_used_for_training_and_denoising": True,
        "clean_reference_used_for_training": False,
        "clean_reference_use": "exclusivamente MSE y PSNR",
        "noise_clipping": False,
        "reconstruction_clipping": False,
        "sigma": sigma,
        "noise_seed": noise_seed,
        "noise_empirical_std": float(noise.std()),
        "patch_size": int(cfg["patch_size"]),
        "stride": int(cfg["stride"]),
        "n_patches_total": patches["n_patches_total"],
        "n_train_patches": int(X.shape[1]),
        "n_atoms": n_atoms,
        "n_iter": n_iter,
        "train_sparsity": train_sparsity,
        "eval_sparsities": [int(x) for x in cfg["eval_sparsities"]],
        "model_seed": model_seed,
        "training_time_s": training_time_s,
        "train_mse_init_dictionary": initial_mse,
        "train_mse_final_recoded": checks["train_mse_final"],
        "checks": checks,
        "controlled_differences_from_external_variant": [
            "origen de los parches de entrenamiento",
            "número de parches disponible (todos los parches válidos de coins ruidosa)",
        ],
        "differences_from_full_paper_estimator": [
            "T0 fijo en lugar de parada OMP por error dependiente de sigma",
            "128 átomos en lugar de 256",
            "paso 4 en lugar de todos los parches solapados",
            "promedio de parches sin término adicional de fidelidad a la observación",
        ],
    }
    (out / "config_adaptativa.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "D": D,
        "D_init": D_init,
        "A_train": A_train,
        "history": hist,
        "noisy": noisy,
        "noise": noise,
        "training_patches": patches,
        "meta": meta,
    }


def build_comparison_table(external_df, adaptive_df):
    """Combina ambas evaluaciones y conserva una sola fila de referencia ruidosa."""
    import pandas as pd

    ext = external_df.copy()
    ada = adaptive_df.copy()
    ext_base = ext[ext["condition"] == "ruidosa_sin_procesar"]
    ada_base = ada[ada["condition"] == "ruidosa_sin_procesar"]
    if len(ext_base) != 1 or len(ada_base) != 1:
        raise ValueError("Cada evaluación debe incluir exactamente una referencia ruidosa")
    for col in ("mse", "psnr_db", "sigma", "noise_seed"):
        if not np.allclose(ext_base[col].to_numpy(), ada_base[col].to_numpy(), equal_nan=True):
            raise AssertionError(f"La referencia ruidosa difiere entre rutas en {col}")

    base = ext_base.copy()
    base["dictionary_source"] = "ninguno"
    base["training_image"] = "ninguna"
    base["method"] = "identidad"
    ext = ext[ext["T0"] > 0].copy()
    ext["dictionary_source"] = "externo_camera_limpia"
    ext["training_image"] = "camera limpia"
    ext["method"] = "ksvd_externo"
    ada = ada[ada["T0"] > 0].copy()
    ada["dictionary_source"] = "adaptativo_coins_ruidosa"
    ada["training_image"] = "coins ruidosa"
    ada["method"] = "ksvd_adaptativo"
    return pd.concat([base, ext, ada], ignore_index=True)


def _write_comparison_note(df, meta: dict, path: Path) -> None:
    noisy = df[df["condition"] == "ruidosa"].copy()
    best = noisy.loc[noisy.groupby("dictionary_source")["psnr_db"].idxmax()]
    lines = [
        "# Comparación de las dos rutas K-SVD",
        "",
        "Se evaluaron dos estrategias con la misma realización de ruido, valores de T0, métricas y "
        "ensamblado: un diccionario externo aprendido con `camera` limpia y un diccionario "
        "adaptativo aprendido directamente de `coins` ruidosa. La segunda estrategia sigue la idea "
        "de la Sec. III-B de Elad y Aharon (2006); la referencia limpia no participa en su "
        "entrenamiento.",
        "",
        "## Mejores resultados sobre la entrada ruidosa",
        "",
        "| diccionario | T0 | MSE | PSNR (dB) |",
        "|---|---:|---:|---:|",
    ]
    for row in best.sort_values("dictionary_source").itertuples():
        lines.append(
            f"| {row.dictionary_source} | {int(row.T0)} | {row.mse:.6f} | {row.psnr_db:.2f} |"
        )
    lines += [
        "",
        "## Alcance",
        "",
        f"El diccionario adaptativo usó {meta['n_train_patches']} parches de la misma observación "
        f"ruidosa, K={meta['n_atoms']} y {meta['n_iter']} iteraciones. La comparación es "
        "transductiva: adaptar el diccionario a la entrada es parte del método y no equivale a "
        "validación sobre una imagen no vista. Se conservó el protocolo reducido del proyecto para "
        "aislar el origen del diccionario; `config_adaptativa.json` enumera las diferencias frente al "
        "estimador completo del artículo.",
        "",
        "## Productos",
        "",
        "- `externo/`: evaluación nueva del modelo de `02_modelo/` bajo la misma ejecución de tiempos.",
        "- `adaptativo/`: modelo, historial, configuración, evaluación y reconstrucciones de la ruta adaptativa.",
        "- `comparacion_resultados.csv`: referencia única y los seis casos de cada diccionario.",
        "- `verificacion_comparacion.json`: comprobaciones de ruido, T0, forma y normas del diccionario.",
        "- `figuras/`: diccionarios, curvas de PSNR y reconstrucciones comparables.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_comparative(root, n_timing_repeats: int = 5, verbose: bool = False) -> dict:
    """Genera la variante adaptativa y una comparación reproducible con la ruta externa."""
    from . import visualizacion as viz

    root = Path(root)
    data_path = root / "01_datos" / "datos.npz"
    config_path = root / "01_datos" / "config.json"
    external_model_path = root / "02_modelo" / "modelo.npz"
    for required in (data_path, config_path, external_model_path):
        if not required.exists():
            raise FileNotFoundError(f"Falta {required}; ejecute primero python run_all.py")

    out = root / "05_comparacion"
    external_dir = out / "externo"
    adaptive_dir = out / "adaptativo"
    external_df = run_experiments(
        data_path,
        external_model_path,
        config_path,
        external_dir,
        n_timing_repeats=n_timing_repeats,
    )
    adaptive = train_adaptive_from_files(data_path, config_path, adaptive_dir, verbose=verbose)
    adaptive_df = run_experiments(
        data_path,
        adaptive_dir / "modelo_adaptativo.npz",
        config_path,
        adaptive_dir,
        n_timing_repeats=n_timing_repeats,
    )
    comparison = build_comparison_table(external_df, adaptive_df)
    comparison.to_csv(out / "comparacion_resultados.csv", index=False)

    with np.load(external_model_path) as model:
        D_external = model["D"].astype(np.float64)
    with np.load(adaptive_dir / "reconstrucciones.npz") as rec_file:
        rec_adaptive = {k: rec_file[k] for k in rec_file.files}
    with np.load(external_dir / "reconstrucciones.npz") as rec_file:
        rec_external = {k: rec_file[k] for k in rec_file.files}
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    fig = out / "figuras"
    viz.plot_dictionary_sources(
        D_external, adaptive["D"], fig / "diccionarios_externo_adaptativo.png"
    )
    viz.plot_psnr_dictionary_comparison(comparison, fig / "comparacion_psnr.png")
    viz.plot_best_dictionary_reconstructions(
        comparison,
        rec_external,
        rec_adaptive,
        fig / "mejores_reconstrucciones.png",
    )
    viz.plot_reconstructions(
        rec_adaptive,
        adaptive_df,
        cfg["eval_sparsities"],
        fig / "reconstrucciones_adaptativas.png",
    )

    same_noisy = bool(np.array_equal(rec_external["noisy_image"], rec_adaptive["noisy_image"]))
    checks = {
        "same_noisy_realization": same_noisy,
        "same_eval_sparsities": sorted(external_df.loc[external_df.T0 > 0, "T0"].unique().tolist())
        == sorted(adaptive_df.loc[adaptive_df.T0 > 0, "T0"].unique().tolist()),
        "adaptive_dictionary_shape": list(adaptive["D"].shape),
        "adaptive_dictionary_unit_norm_max_dev": float(
            np.max(np.abs(np.linalg.norm(adaptive["D"], axis=0) - 1.0))
        ),
        "adaptive_training_uses_clean_reference": False,
        "comparison_rows": int(len(comparison)),
    }
    if not checks["same_noisy_realization"] or not checks["same_eval_sparsities"]:
        raise AssertionError("Las rutas no comparten el protocolo de evaluación")
    if checks["adaptive_dictionary_unit_norm_max_dev"] >= 1e-6:
        raise AssertionError("El diccionario adaptativo no tiene columnas unitarias")
    (out / "verificacion_comparacion.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_comparison_note(comparison, adaptive["meta"], out / "README.md")
    return {"adaptive": adaptive, "adaptive_results": adaptive_df, "comparison": comparison,
            "checks": checks}
