import warnings

import numpy as np
import pytest

from src.modelo import (count_active, fit_ksvd, init_dictionary, ksvd_dictionary_update,
                        normalize_columns, rank1_atom_update, sparse_code, validate_model)


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    D_true = normalize_columns(rng.standard_normal((64, 96)))
    A = np.zeros((96, 400))
    for i in range(400):
        idx = rng.choice(96, 3, replace=False)
        A[idx, i] = rng.standard_normal(3)
    X = D_true @ A + 0.01 * rng.standard_normal((64, 400))
    return X - X.mean(axis=0)


def test_sparse_code_shapes_and_max_nonzero(synthetic):
    D = init_dictionary(synthetic, 128, 1)
    A = sparse_code(D, synthetic, 4)
    assert A.shape == (128, synthetic.shape[1])
    assert count_active(A).max() <= 4


def test_sparse_code_single_signal_is_2d(synthetic):
    D = init_dictionary(synthetic, 128, 1)
    assert sparse_code(D, synthetic[:, 0], 3).shape == (128, 1)
    assert sparse_code(D, synthetic[:, :1], 3).shape == (128, 1)


def test_sparse_code_zero_signals_get_zero_codes(synthetic):
    D = init_dictionary(synthetic, 128, 1)
    Z = synthetic[:, :5].copy(); Z[:, 2] = 0.0; Z[:, 4] = 1e-12
    A = sparse_code(D, Z, 4)
    assert np.all(A[:, 2] == 0) and np.all(A[:, 4] == 0) and np.any(A[:, 0] != 0)


def test_sparse_code_matches_sklearn_directly(synthetic):
    from sklearn.linear_model import orthogonal_mp
    D = init_dictionary(synthetic, 128, 1)
    A = sparse_code(D, synthetic[:, :50], 4)
    np.testing.assert_allclose(A, orthogonal_mp(D, synthetic[:, :50], n_nonzero_coefs=4), atol=1e-10)


def test_sparse_code_rejects_unnormalized(synthetic):
    with pytest.raises(ValueError):
        sparse_code(2 * init_dictionary(synthetic, 128, 1), synthetic, 4)


def test_sparse_code_reports_premature_stop():
    D = np.eye(64)
    z = D[:, 0] * 3.0
    with pytest.warns(RuntimeWarning, match="máximo"):
        A = sparse_code(D, z, 4)
    assert count_active(A).max() == 1


def test_rank1_svd_is_best_rank1_approximation():
    """Actualización de soporte fijo: (d, a) de la SVD minimiza ||E - d a^T||_F."""
    rng = np.random.default_rng(1)
    E = rng.standard_normal((64, 30))
    d, a = rank1_atom_update(E)
    s = np.linalg.svd(E, compute_uv=False)
    err = np.linalg.norm(E - np.outer(d, a))
    assert abs(np.linalg.norm(d) - 1) < 1e-12
    assert abs(err ** 2 - np.sum(s[1:] ** 2)) < 1e-9          # Eckart-Young
    for _ in range(200):                                      # ninguna alternativa rank-1 es mejor
        u = rng.standard_normal(64); u /= np.linalg.norm(u)
        best_v = E.T @ u                                       # coeficientes óptimos dado u
        assert np.linalg.norm(E - np.outer(u, best_v)) >= err - 1e-12


def test_ksvd_update_restricted_support_and_monotone(synthetic):
    X = synthetic
    D = init_dictionary(X, 128, 3)
    A = sparse_code(D, X, 4)
    support_before = A != 0
    err_before = np.linalg.norm(X - D @ A)
    D2, A2, info = ksvd_dictionary_update(X, D, A)
    assert not np.any((A2 != 0) & ~support_before)            # soporte no crece
    np.testing.assert_allclose(np.linalg.norm(D2, axis=0), 1, atol=1e-12)
    assert np.linalg.norm(X - D2 @ A2) <= err_before + 1e-10  # soporte fijo: no aumenta


def test_ksvd_update_matches_literal_formula_for_first_atom(synthetic):
    X = synthetic
    D = init_dictionary(X, 128, 3)
    A = sparse_code(D, X, 4)
    j = int(np.argmax(np.sum(A != 0, axis=1)))
    # reordenar para que el átomo j sea el primero visitado
    perm = np.r_[j, np.delete(np.arange(128), j)]
    Dp, Ap = D[:, perm], A[perm]
    omega = np.flatnonzero(Ap[0] != 0)
    E = X[:, omega] - Dp @ Ap[:, omega] + np.outer(Dp[:, 0], Ap[0, omega])
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    D2, A2, _ = ksvd_dictionary_update(X, Dp, Ap)
    np.testing.assert_allclose(np.outer(D2[:, 0], A2[0, omega]), s[0] * np.outer(U[:, 0], Vt[0]), atol=1e-12)


def test_unused_atom_is_reinitialized_with_normalized_residual(synthetic):
    X = synthetic
    D = init_dictionary(X, 128, 3)
    A = sparse_code(D, X, 4)
    A[0, :] = 0.0                                             # átomo 0 sin uso (se visita primero)
    R = X - D @ A
    rn = np.linalg.norm(R, axis=0)
    i = int(np.argmax(rn))
    D2, A2, info = ksvd_dictionary_update(X, D, A)
    assert info["unused_atoms"] >= 1 and info["reinitialized_atoms"] >= 1
    np.testing.assert_allclose(D2[:, 0], R[:, i] / rn[i], atol=1e-12)
    assert np.all(A2[0] == 0)


def test_unused_atom_kept_when_residual_is_zero():
    D = np.eye(8)[:, :4]
    X = D[:, :2] * 2.0                                        # perfectamente representable
    A = np.zeros((4, 2)); A[0, 0] = 2.0; A[1, 1] = 2.0
    D2, _, info = ksvd_dictionary_update(X, D, A)
    assert info["unused_atoms"] == 2 and info["reinitialized_atoms"] == 0
    np.testing.assert_array_equal(D2[:, 2:], D[:, 2:])


def test_fit_ksvd_contract_and_determinism(synthetic):
    D, A, hist = fit_ksvd(synthetic, n_atoms=128, n_iter=3, train_sparsity=4, seed=43)
    D_b, A_b, _ = fit_ksvd(synthetic, n_atoms=128, n_iter=3, train_sparsity=4, seed=43)
    np.testing.assert_allclose(D, D_b, atol=1e-12)
    checks = validate_model(D, A, synthetic, 4)
    assert checks["max_active_per_column"] <= 4
    assert [h["iteration"] for h in hist] == [1, 2, 3]
    assert all(np.isfinite(h["train_mse"]) for h in hist)
    # A_train es el recodificado con el D final
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.testing.assert_allclose(A, sparse_code(D, synthetic, 4), atol=1e-12)
    # mejora frente al diccionario inicial
    D0 = init_dictionary(synthetic, 128, 43)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mse0 = np.mean((synthetic - D0 @ sparse_code(D0, synthetic, 4)) ** 2)
    assert np.mean((synthetic - D @ A) ** 2) < mse0


def test_init_dictionary_uses_nonconstant_columns():
    X = np.zeros((64, 200)); X[:, 100:] = np.random.default_rng(0).standard_normal((64, 100))
    D = init_dictionary(X, 50, 43)
    np.testing.assert_allclose(np.linalg.norm(D, axis=0), 1)
    with pytest.raises(ValueError):
        init_dictionary(X, 101, 43)
