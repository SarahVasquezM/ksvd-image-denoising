import json

import numpy as np
import pandas as pd

from src.adaptativo import (
    build_comparison_table,
    extract_adaptive_training_patches,
    train_adaptive_from_files,
)
from src.evaluacion import make_noisy
from src.preprocesamiento import extract_patches


def test_adaptive_patches_come_from_noisy_observation_only():
    clean = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float64).reshape(32, 32)
    noisy, _ = make_noisy(clean, 20 / 255, 44)
    got = extract_adaptive_training_patches(noisy, patch_size=8, stride=4)
    expected, _, _ = extract_patches(noisy, patch_size=8, stride=4)
    np.testing.assert_array_equal(got["X_train"], expected[:, got["indices"]])
    assert got["X_train"].dtype == np.float64
    assert got["n_patches_total"] == expected.shape[1]


def test_adaptive_training_contract_on_small_image(tmp_path):
    clean = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float64).reshape(32, 32)
    data_path = tmp_path / "datos.npz"
    config_path = tmp_path / "config.json"
    out = tmp_path / "adaptativo"
    np.savez(data_path, test_image=clean)
    cfg = {
        "patch_size": 8,
        "stride": 4,
        "min_patch_norm": 1e-8,
        "n_atoms": 16,
        "n_iter": 2,
        "train_sparsity": 2,
        "eval_sparsities": [1, 2],
        "sigma": 20 / 255,
        "seeds": {"model": 43, "ksvd": 43, "noise": 44},
    }
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    result = train_adaptive_from_files(data_path, config_path, out)
    assert result["D"].shape == (64, 16)
    np.testing.assert_allclose(np.linalg.norm(result["D"], axis=0), 1.0, atol=1e-10)
    assert result["meta"]["clean_reference_used_for_training"] is False
    expected_noisy, _ = make_noisy(clean, cfg["sigma"], cfg["seeds"]["noise"])
    expected, _, _ = extract_patches(expected_noisy, 8, 4)
    idx = result["training_patches"]["indices"]
    np.testing.assert_array_equal(result["training_patches"]["X_train"], expected[:, idx])
    assert (out / "modelo_adaptativo.npz").exists()
    assert (out / "config_adaptativa.json").exists()


def test_comparison_table_has_one_baseline_and_both_dictionary_sources():
    columns = {
        "case_id": ["noisy_baseline", "clean_t2", "noisy_t2"],
        "input_kind": ["noisy", "clean", "noisy"],
        "method": ["identity", "ksvd", "ksvd"],
        "sigma": [0.1, 0.1, 0.1],
        "condition": ["ruidosa_sin_procesar", "limpia", "ruidosa"],
        "T0": [0, 2, 2],
        "mse": [0.01, 0.002, 0.003],
        "psnr_db": [20.0, 27.0, 25.0],
        "noise_seed": [44, 44, 44],
    }
    external = pd.DataFrame(columns)
    adaptive = external.copy()
    adaptive.loc[adaptive["condition"] == "ruidosa", "psnr_db"] = 25.5
    got = build_comparison_table(external, adaptive)
    assert (got["condition"] == "ruidosa_sin_procesar").sum() == 1
    assert set(got["dictionary_source"]) == {
        "ninguno",
        "externo_camera_limpia",
        "adaptativo_coins_ruidosa",
    }
