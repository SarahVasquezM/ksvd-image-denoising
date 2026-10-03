"""Orquestación de los cuatro bloques. Cada función lee los productos del bloque anterior desde
disco (contrato de archivos) y escribe los suyos en su carpeta.

Uso:  python -m src.pipeline [--blocks 1 2 3 4]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import visualizacion as viz
from .evaluacion import run_experiments
from .modelo import load_model, sparse_code, train_from_files
from .preprocesamiento import build_dataset, extract_patches, load_config, load_dataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THREADS = 4


def paths(root=ROOT) -> dict:
    root = Path(root)
    return {
        "root": root,
        "datos_dir": root / "01_datos", "modelo_dir": root / "02_modelo",
        "eval_dir": root / "03_evaluacion", "final_dir": root / "04_final",
        "datos": root / "01_datos" / "datos.npz", "config": root / "01_datos" / "config.json",
        "modelo": root / "02_modelo" / "modelo.npz", "historial": root / "02_modelo" / "historial.csv",
        "config_efectiva": root / "02_modelo" / "config_efectiva.json",
        "resultados": root / "03_evaluacion" / "resultados.csv",
        "reconstrucciones": root / "03_evaluacion" / "reconstrucciones.npz",
    }


def run_block1(root=ROOT) -> dict:
    p = paths(root)
    res = build_dataset(p["datos_dir"])
    d = load_dataset(p["datos"])
    cfg = res["config"]
    _, _, positions = extract_patches(d["train_image"], cfg["patch_size"], cfg["stride"])
    fig = p["datos_dir"] / "figuras"
    viz.plot_images(d["train_image"], d["test_image"], fig / "imagenes_camera_coins.png")
    viz.plot_patch_sample(d["train_image"], d["X_train"], d["train_indices"], positions,
                          fig / "muestra_parches.png", patch_size=cfg["patch_size"])
    return res


def run_block2(root=ROOT, time_budget_s: float = 3600.0) -> dict:
    import pandas as pd
    p = paths(root)
    res = train_from_files(p["datos"], p["config"], p["modelo_dir"], time_budget_s=time_budget_s)
    hist = pd.read_csv(p["historial"])
    fig = p["modelo_dir"] / "figuras"
    viz.plot_dictionaries_side_by_side(res["D_init"], res["D"], fig / "diccionario_inicial_vs_final.png")
    viz.plot_dictionary(res["D"], fig / "diccionario_final.png", "Diccionario final K-SVD")
    viz.plot_training_curve(hist, fig / "curva_entrenamiento.png",
                            init_mse=res["effective"]["train_mse_init_dictionary"])
    viz.plot_atom_usage(res["A_train"], fig / "uso_atomos.png")
    return res


def run_block3(root=ROOT):
    p = paths(root)
    df = run_experiments(p["datos"], p["modelo"], p["config"], p["eval_dir"])
    cfg = load_config(p["config"])
    with np.load(p["reconstrucciones"]) as r:
        rec = {k: r[k] for k in r.files}
    fig = p["eval_dir"] / "figuras"
    T = cfg["eval_sparsities"]
    viz.plot_reconstructions(rec, df, T, fig / "reconstrucciones.png")
    viz.plot_error_maps(rec, T, fig / "mapas_error.png")
    viz.plot_psnr_vs_t0(df, fig / "psnr_vs_t0.png")
    viz.plot_mse_vs_t0(df, fig / "mse_vs_t0.png")
    viz.plot_activity_time(df, fig / "actividad_tiempo.png")
    return df


def run_block4(root=ROOT) -> dict:
    """Integración: verifica coherencia entre bloques y genera la figura-resumen y tabla final."""
    import pandas as pd
    p = paths(root)
    cfg = load_config(p["config"])
    eff = json.loads(Path(p["config_efectiva"]).read_text(encoding="utf-8"))
    d = load_dataset(p["datos"])
    m = load_model(p["modelo"])
    hist = pd.read_csv(p["historial"])
    df = pd.read_csv(p["resultados"])
    with np.load(p["reconstrucciones"]) as r:
        rec = {k: r[k] for k in r.files}

    K = m["D"].shape[1]
    n_used = eff["n_train_patches"]
    A_re = sparse_code(m["D"], d["X_train"][:, :n_used], cfg["train_sparsity"])
    checks = {
        "D_shape": list(m["D"].shape),
        "K_matches_effective_config": K == eff["n_atoms"],
        "K_matches_results": bool((df["n_atoms"] == K).all()),
        "A_train_reproducible": bool(np.allclose(A_re, m["A_train"], atol=1e-10)),
        "history_rows_match_n_iter": len(hist) == eff["n_iter"],
        "test_image_matches": bool(np.array_equal(rec["test_image"], d["test_image"])),
        "eval_sparsities_present": sorted(df.loc[df.T0 > 0, "T0"].unique().tolist()) == sorted(cfg["eval_sparsities"]),
        "contingency_level": eff["contingency_level"],
    }
    for k, v in checks.items():
        if isinstance(v, bool) and not v:
            raise AssertionError(f"Fallo de integración: {k}")

    fdir = p["final_dir"] / "figuras"
    viz.plot_summary(m["D"], hist, df, rec, cfg["eval_sparsities"], fdir / "resumen_resultados.png")
    viz.plot_reconstructions(rec, df, cfg["eval_sparsities"], fdir / "reconstrucciones.png")
    viz.plot_psnr_vs_t0(df, fdir / "psnr_vs_t0.png")
    viz.plot_dictionaries_side_by_side(m["D_init"], m["D"], fdir / "diccionarios.png")
    viz.plot_training_curve(hist, fdir / "curva_entrenamiento.png", init_mse=eff["train_mse_init_dictionary"])
    viz.plot_activity_time(df, fdir / "actividad_tiempo.png")

    table = df.copy()
    table.to_csv(p["final_dir"] / "tabla_final.csv", index=False)
    (p["final_dir"] / "verificacion_integracion.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"checks": checks, "table": table}


def main(blocks=(1, 2, 3, 4), root=ROOT, threads: int | None = DEFAULT_THREADS):
    """Ejecuta los bloques pedidos. ``threads`` limita los hilos BLAS/OpenMP (None = sin límite).

    Con matrices pequeñas (64 x ~1500) muchos hilos BLAS empeoran el tiempo por sobresuscripción;
    por defecto se usan 4 hilos (los de una sesión CPU de Kaggle).
    """
    from contextlib import nullcontext
    from threadpoolctl import threadpool_limits
    ctx = threadpool_limits(limits=threads) if threads else nullcontext()
    with ctx:
        return _main(blocks, root)


def _main(blocks, root):
    out = {}
    if 1 in blocks:
        out[1] = run_block1(root); print("[1] datos:", json.dumps(out[1]["summary"]))
    if 2 in blocks:
        out[2] = run_block2(root); print("[2] modelo: t =", f"{out[2]['effective']['training_time_s']:.2f}s")
    if 3 in blocks:
        out[3] = run_block3(root); print("[3] evaluación:\n", out[3].to_string(index=False))
    if 4 in blocks:
        out[4] = run_block4(root); print("[4] integración:", json.dumps(out[4]["checks"]))
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                    help="límite de hilos BLAS (0 = sin límite)")
    a = ap.parse_args()
    main(tuple(a.blocks), threads=a.threads or None)
