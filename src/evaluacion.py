"""Bloque 3 - Evaluación: reconstrucción de la imagen de prueba limpia y ruidosa.

Para cada T0 en ``eval_sparsities`` y cada condición (limpia / ruidosa):
  1. extraer parches centrados de la imagen de entrada;
  2. codificar con OMP (a lo más T0 átomos) sobre el diccionario aprendido;
  3. restaurar la media de cada parche y reensamblar por promedio de solapamientos;
  4. comparar SIEMPRE contra la imagen de prueba limpia (MSE, PSNR con data_range=1).
Se usa UNA sola realización de ruido (RNG 44) para todos los T0.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

from .modelo import count_active, sparse_code
from .preprocesamiento import assemble_patches, extract_patches

N_TIMING_REPEATS = 5


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def mse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(mean_squared_error(reference, estimate))


def psnr(reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0) -> float:
    """PSNR en dB vía ``skimage.metrics.peak_signal_noise_ratio``; ``inf`` si MSE = 0."""
    if mse(reference, estimate) == 0.0:
        return float("inf")
    return float(peak_signal_noise_ratio(reference, estimate, data_range=data_range))


def metric_sanity_checks(image: np.ndarray, data_range: float = 1.0) -> dict:
    """Comprobaciones previas obligatorias de las métricas."""
    shifted = image + 0.1
    res = {
        "self_mse": mse(image, image),
        "self_psnr": psnr(image, image, data_range),
        "const_0.1_mse": mse(image, shifted),
        "const_0.1_psnr": psnr(image, shifted, data_range),
    }
    assert res["self_mse"] == 0.0
    assert np.isinf(res["self_psnr"]) and res["self_psnr"] > 0
    assert abs(res["const_0.1_mse"] - 0.01) < 1e-12
    assert abs(res["const_0.1_psnr"] - 20.0) < 1e-9
    return res


# ---------------------------------------------------------------------------
# Ruido y reconstrucción
# ---------------------------------------------------------------------------
def make_noisy(image: np.ndarray, sigma: float, seed: int):
    """Ruido gaussiano aditivo (sin clipping). Devuelve ``(noisy, noise)``."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=image.shape)
    return image + noise, noise


def reconstruct_image(image: np.ndarray, D: np.ndarray, max_nonzero: int, patch_size: int = 8,
                      stride: int = 4) -> dict:
    """Reconstruye ``image`` con el diccionario ``D`` (OMP con a lo más ``max_nonzero`` átomos)."""
    t0 = time.perf_counter()
    Z, means, positions = extract_patches(image, patch_size, stride)
    t1 = time.perf_counter()
    A = sparse_code(D, Z, max_nonzero)
    t2 = time.perf_counter()
    patches = D @ A + means[None, :]
    rec = assemble_patches(patches, positions, image.shape, patch_size)
    t3 = time.perf_counter()
    return {"image": rec, "A": A, "n_patches": Z.shape[1],
            "time_total_s": t3 - t0, "time_coding_s": t2 - t1}


def run_experiments(data_path, model_path, config_path, output_dir,
                    n_timing_repeats: int = N_TIMING_REPEATS) -> "pd.DataFrame":
    """Ejecuta el experimento completo y escribe ``resultados.csv`` y ``reconstrucciones.npz``.

    Lee la configuración (patch_size, stride, sigma, semillas, T0) y las formas (K, n) de los
    archivos reales. El tiempo reportado es la mediana de ``n_timing_repeats`` ejecuciones
    idénticas (extracción + OMP + reensamblado).
    """
    import pandas as pd

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    with np.load(data_path) as d:
        test = d["test_image"].astype(np.float64)
    with np.load(model_path) as m:
        D = m["D"].astype(np.float64)
    p, s = int(cfg["patch_size"]), int(cfg["stride"])
    n, K = D.shape
    if n != p * p:
        raise ValueError(f"Dimensión de átomo {n} incompatible con patch_size={p}")
    sigma = float(cfg["sigma"])
    data_range = float(cfg.get("data_range", 1.0))
    sparsities = [int(t) for t in cfg["eval_sparsities"]]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    checks = metric_sanity_checks(test, data_range)
    noisy, noise = make_noisy(test, sigma, int(cfg["seeds"]["noise"]))
    inputs = {"limpia": test, "ruidosa": noisy}

    rows = [{
        "condition": "ruidosa_sin_procesar", "T0": 0,
        "mse": mse(test, noisy), "psnr_db": psnr(test, noisy, data_range),
        "mean_active_coefs": np.nan, "max_active_coefs": np.nan,
        "reconstruction_time_s": np.nan, "coding_time_s": np.nan,
        "n_patches": np.nan, "n_atoms": K,
    }]
    arrays = {
        "original": test,
        "noisy": noisy,
        "test_image": test,          # alias de compatibilidad con la integración existente
        "noisy_image": noisy,
        "noise": noise,
    }
    for cond, img in inputs.items():
        for T0 in sparsities:
            runs = [reconstruct_image(img, D, T0, p, s) for _ in range(max(1, n_timing_repeats))]
            r = runs[0]
            for other in runs[1:]:
                if not np.array_equal(other["image"], r["image"]):
                    raise AssertionError("Reconstrucción no determinista")
            act = count_active(r["A"])
            if act.max() > T0:
                raise AssertionError(f"Más de T0={T0} coeficientes activos")
            rows.append({
                "condition": cond, "T0": T0,
                "mse": mse(test, r["image"]), "psnr_db": psnr(test, r["image"], data_range),
                "mean_active_coefs": float(act.mean()), "max_active_coefs": int(act.max()),
                "reconstruction_time_s": float(np.median([x["time_total_s"] for x in runs])),
                "coding_time_s": float(np.median([x["time_coding_s"] for x in runs])),
                "n_patches": r["n_patches"], "n_atoms": K,
            })
            arrays[f"rec_{cond}_T{T0}"] = r["image"]
            contract_prefix = "clean" if cond == "limpia" else "noisy"
            arrays[f"{contract_prefix}_t{T0}"] = r["image"]

    df = pd.DataFrame(rows)
    case_ids = {("limpia", t): f"clean_t{t}" for t in sparsities}
    case_ids.update({("ruidosa", t): f"noisy_t{t}" for t in sparsities})
    df.insert(0, "case_id", [
        "noisy_baseline" if c == "ruidosa_sin_procesar" else case_ids[(c, int(t))]
        for c, t in zip(df["condition"], df["T0"])
    ])
    df.insert(1, "input_kind", ["clean" if c == "limpia" else "noisy" for c in df["condition"]])
    df.insert(2, "method", ["identity" if c == "ruidosa_sin_procesar" else "ksvd"
                             for c in df["condition"]])
    df.insert(3, "sigma", sigma)
    df["max_nonzero"] = df["T0"].astype(int)
    df["mean_nonzero"] = df["mean_active_coefs"]
    df["reconstruction_s"] = df["reconstruction_time_s"]
    df["noise_seed"] = int(cfg["seeds"]["noise"])
    df.to_csv(out / "resultados.csv", index=False)
    np.savez(out / "reconstrucciones.npz", **arrays)
    meta = {
        "metric_checks": checks,
        "sigma": sigma, "noise_seed": int(cfg["seeds"]["noise"]),
        "noise_realizations": 1, "noise_clipping": False,
        "noise_empirical_std": float(noise.std()),
        "reconstruction_clipping": False,
        "reference_for_metrics": "imagen de prueba limpia (coins 128x128)",
        "n_timing_repeats": n_timing_repeats,
        "timing": "mediana de repeticiones; extracción + OMP + reensamblado",
        "n_atoms": K, "atom_dim": n,
    }
    with open(out / "evaluacion_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return df


if __name__ == "__main__":  # pragma: no cover
    import argparse
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Evalúa el diccionario en la imagen de prueba")
    ap.add_argument("--data", default=str(root / "01_datos" / "datos.npz"))
    ap.add_argument("--model", default=str(root / "02_modelo" / "modelo.npz"))
    ap.add_argument("--config", default=str(root / "01_datos" / "config.json"))
    ap.add_argument("--output-dir", default=str(root / "03_evaluacion"))
    args = ap.parse_args()
    print(run_experiments(args.data, args.model, args.config, args.output_dir).to_string(index=False))
