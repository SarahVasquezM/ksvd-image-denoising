"""Integración: (a) los artefactos versionados cumplen el contrato; (b) el pipeline completo
corre en un directorio temporal y reproduce los resultados versionados."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts():
    d = np.load(ROOT / "01_datos" / "datos.npz")
    m = np.load(ROOT / "02_modelo" / "modelo.npz")
    cfg = json.loads((ROOT / "01_datos" / "config.json").read_text(encoding="utf-8"))
    eff = json.loads((ROOT / "02_modelo" / "config_efectiva.json").read_text(encoding="utf-8"))
    hist = pd.read_csv(ROOT / "02_modelo" / "historial.csv")
    res = pd.read_csv(ROOT / "03_evaluacion" / "resultados.csv")
    rec = np.load(ROOT / "03_evaluacion" / "reconstrucciones.npz")
    return d, m, cfg, eff, hist, res, rec


def test_datos_contract(artifacts):
    d, _, cfg, *_ = artifacts
    assert d["train_image"].shape == (256, 256) and d["test_image"].shape == (128, 128)
    assert d["X_train"].shape == (64, 1500) and d["train_indices"].shape == (1500,)
    for k in ("train_image", "test_image", "X_train"):
        assert d[k].dtype == np.float64
    assert np.issubdtype(d["train_indices"].dtype, np.integer)
    for key in ("version_contrato", "patch_size", "stride", "n_atoms", "n_iter", "train_sparsity",
                "eval_sparsities", "sigma", "seeds", "train_image_name", "test_image_name", "dimensions"):
        assert key in cfg
    from src.preprocesamiento import extract_patches
    Z, _, _ = extract_patches(d["train_image"], cfg["patch_size"], cfg["stride"])
    np.testing.assert_array_equal(d["X_train"], Z[:, d["train_indices"]])


def test_modelo_contract(artifacts):
    d, m, cfg, eff, hist, *_ = artifacts
    K = eff["n_atoms"]
    assert m["D"].shape == (64, K) and m["D_init"].shape == (64, K)
    assert m["A_train"].shape == (K, eff["n_train_patches"])
    np.testing.assert_allclose(np.linalg.norm(m["D"], axis=0), 1, atol=1e-10)
    assert np.max(np.sum(np.abs(m["A_train"]) > 1e-10, axis=0)) <= cfg["train_sparsity"]
    assert list(hist.columns) == ["iteration", "train_mse", "elapsed_s", "unused_atoms"]
    assert len(hist) == eff["n_iter"]
    if eff["contingency_level"] == 0:
        assert (K, eff["n_train_patches"], eff["n_iter"]) == (128, 1500, 8)


def test_resultados_contract(artifacts):
    _, m, cfg, _, _, res, rec = artifacts
    required = {"case_id", "input_kind", "method", "sigma", "max_nonzero", "mse", "psnr_db",
                "mean_nonzero", "reconstruction_s", "noise_seed"}
    assert required <= set(res.columns)
    assert set(res.case_id) == {"noisy_baseline", "clean_t2", "clean_t4", "clean_t8",
                                "noisy_t2", "noisy_t4", "noisy_t8"}
    assert set(res.loc[res.T0 > 0, "T0"]) == set(cfg["eval_sparsities"])
    assert set(res.condition) == {"limpia", "ruidosa", "ruidosa_sin_procesar"}
    for T0 in cfg["eval_sparsities"]:
        for c in ("limpia", "ruidosa"):
            row = res[(res.condition == c) & (res.T0 == T0)].iloc[0]
            img = rec[f"rec_{c}_T{T0}"]
            assert row.mse == pytest.approx(np.mean((img - rec["test_image"]) ** 2), rel=1e-12)
            assert row.mean_active_coefs <= T0
    from src.evaluacion import make_noisy
    noisy, _ = make_noisy(rec["test_image"], cfg["sigma"], cfg["seeds"]["noise"])
    np.testing.assert_array_equal(noisy, rec["noisy_image"])
    np.testing.assert_array_equal(rec["original"], rec["test_image"])
    np.testing.assert_array_equal(rec["noisy"], rec["noisy_image"])
    for T0 in cfg["eval_sparsities"]:
        np.testing.assert_array_equal(rec[f"clean_t{T0}"], rec[f"rec_limpia_T{T0}"])
        np.testing.assert_array_equal(rec[f"noisy_t{T0}"], rec[f"rec_ruidosa_T{T0}"])


def test_full_pipeline_reproduces_results(tmp_path):
    """Ejecuta los 4 bloques en una copia limpia y compara con los artefactos versionados."""
    from src.pipeline import main
    for sub in ("01_datos", "02_modelo", "03_evaluacion", "04_final"):
        (tmp_path / sub).mkdir()
    main(blocks=(1, 2, 3, 4), root=tmp_path)
    new = np.load(tmp_path / "01_datos" / "datos.npz")
    old = np.load(ROOT / "01_datos" / "datos.npz")
    for k in old.files:
        np.testing.assert_array_equal(new[k], old[k])
    Dn = np.load(tmp_path / "02_modelo" / "modelo.npz")["D"]
    Do = np.load(ROOT / "02_modelo" / "modelo.npz")["D"]
    np.testing.assert_allclose(Dn, Do, atol=1e-8)
    rn = pd.read_csv(tmp_path / "03_evaluacion" / "resultados.csv")
    ro = pd.read_csv(ROOT / "03_evaluacion" / "resultados.csv")
    np.testing.assert_allclose(rn.psnr_db.replace(np.inf, 0), ro.psnr_db.replace(np.inf, 0), atol=1e-6)
    np.testing.assert_allclose(rn.mse, ro.mse, rtol=1e-6)
    assert json.loads((tmp_path / "04_final" / "verificacion_integracion.json").read_text())["K_matches_results"]
