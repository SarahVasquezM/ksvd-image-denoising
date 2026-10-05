import numpy as np
import pytest

from src.evaluacion import make_noisy, metric_sanity_checks, mse, psnr, reconstruct_image
from src.modelo import init_dictionary, normalize_columns
from src.preprocesamiento import extract_patches, load_images


@pytest.fixture(scope="module")
def test_image():
    return load_images()[1]


def test_metric_identity_and_constant_shift(test_image):
    assert mse(test_image, test_image) == 0.0
    assert psnr(test_image, test_image) == float("inf")
    assert mse(test_image, test_image + 0.1) == pytest.approx(0.01, abs=1e-12)
    assert psnr(test_image, test_image + 0.1) == pytest.approx(20.0, abs=1e-9)
    metric_sanity_checks(test_image)


def test_noise_single_realization_reproducible(test_image):
    n1, z1 = make_noisy(test_image, 20 / 255, 44)
    n2, z2 = make_noisy(test_image, 20 / 255, 44)
    np.testing.assert_array_equal(n1, n2)
    rng = np.random.default_rng(44)
    np.testing.assert_array_equal(z1, rng.normal(0, 20 / 255, size=test_image.shape))
    assert abs(z1.std() - 20 / 255) < 0.005
    np.testing.assert_array_equal(n1, test_image + z1)        # sin clipping


def test_reconstruction_with_complete_basis_is_exact(test_image):
    """Con una base ortonormal completa (64 átomos) y T0=64 la reconstrucción es exacta."""
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((64, 64)))
    r = reconstruct_image(test_image, Q, 64)
    assert np.max(np.abs(r["image"] - test_image)) < 1e-10


def test_reconstruction_respects_t0(test_image):
    Z, _, _ = extract_patches(load_images()[0])
    D = init_dictionary(Z, 128, 43)
    for T0 in (2, 4, 8):
        r = reconstruct_image(test_image, D, T0)
        assert r["image"].shape == test_image.shape
        assert np.max(np.sum(np.abs(r["A"]) > 1e-10, axis=0)) <= T0
        assert r["time_total_s"] > 0
