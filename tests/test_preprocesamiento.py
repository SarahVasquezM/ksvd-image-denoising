import numpy as np
import pytest

from src.preprocesamiento import (DEFAULT_CONFIG, assemble_patches, extract_patches, load_images,
                                  patch_starts, sample_training_patches)


@pytest.fixture(scope="module")
def images():
    return load_images()


def test_image_shapes_and_dtype(images):
    train, test = images
    assert train.shape == (256, 256) and test.shape == (128, 128)
    assert train.dtype == np.float64 and test.dtype == np.float64
    assert 0.0 <= train.min() and train.max() <= 1.0


def test_patch_starts_includes_last_and_no_duplicates():
    assert patch_starts(256, 8, 4)[-1] == 248
    assert patch_starts(10, 8, 4) == [0, 2]          # 0, y el último inicio válido 2
    assert patch_starts(12, 8, 4) == [0, 4]          # sin duplicar 4
    s = patch_starts(13, 8, 4)
    assert s == [0, 4, 5] and len(set(s)) == len(s)


def test_extract_shapes_and_centering(images):
    train, _ = images
    Z, means, pos = extract_patches(train)
    assert Z.shape == (64, 63 * 63) and means.shape == (63 * 63,) and pos.shape == (63 * 63, 2)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-14)
    assert len({tuple(p) for p in pos}) == pos.shape[0]


def test_extract_order_rows_then_cols_flatten_C():
    img = np.arange(12 * 13, dtype=float).reshape(12, 13)
    Z, means, pos = extract_patches(img, 8, 4)
    assert pos.tolist() == [[0, 0], [0, 4], [0, 5], [4, 0], [4, 4], [4, 5]]
    k = 2
    r, c = pos[k]
    np.testing.assert_allclose(Z[:, k] + means[k], img[r:r + 8, c:c + 8].flatten(order="C"))


@pytest.mark.parametrize("which", [0, 1])
def test_roundtrip_mandatory(images, which):
    image = images[which]
    Z, means, positions = extract_patches(image)
    recovered = assemble_patches(Z + means[None, :], positions, image.shape)
    assert np.max(np.abs(recovered - image)) < 1e-10


def test_assemble_no_clipping_and_coverage():
    img = np.linspace(-0.5, 1.5, 20 * 20).reshape(20, 20)   # fuera de [0,1]
    Z, means, pos = extract_patches(img, 8, 4)
    rec, cov = assemble_patches(Z + means, pos, img.shape, return_coverage=True)
    assert rec.min() < 0 and rec.max() > 1 and cov.min() >= 1
    with pytest.raises(ValueError):
        assemble_patches(Z[:, :1] + means[:1], pos[:1], img.shape)   # cobertura incompleta


def test_invalid_inputs_are_rejected():
    for image, p, s in [
        (np.ones((7, 8)), 8, 4),
        (np.ones((8, 8, 1)), 8, 4),
        (np.full((8, 8), np.nan), 8, 4),
        (np.ones((8, 8)), 0, 4),
        (np.ones((8, 8)), 8, 0),
        (np.ones((8, 8)), 8, 9),
        (np.ones((8, 8)), 8.0, 4),
        (np.ones((8, 8)), True, 4),
    ]:
        with pytest.raises(ValueError):
            extract_patches(image, p, s)
    for patches, positions, shape in [
        (np.ones((64, 1)), [[0.0, 0.0]], (8, 8)),
        (np.ones((64, 1)), [[-1, 0]], (8, 8)),
        (np.ones((64, 1)), [[0, 0]], (9, 9)),
        (np.ones((63, 1)), [[0, 0]], (8, 8)),
        (np.full((64, 1), np.inf), [[0, 0]], (8, 8)),
    ]:
        with pytest.raises(ValueError):
            assemble_patches(patches, positions, shape)


def test_sampling_reproducible_and_indices_refer_to_full_extraction(images):
    train, _ = images
    Z, _, _ = extract_patches(train)
    X1, i1 = sample_training_patches(Z, 1500, 42)
    X2, i2 = sample_training_patches(Z, 1500, 42)
    assert np.array_equal(i1, i2) and len(set(i1.tolist())) == 1500
    assert np.array_equal(X1, Z[:, i1])


def test_sampling_filters_flat_patches():
    img = np.zeros((16, 16)); img[8:, 8:] = np.random.default_rng(0).random((8, 8))
    Z, _, _ = extract_patches(img, 8, 4)
    valid = np.flatnonzero(np.linalg.norm(Z, axis=0) > 1e-8)
    _, idx = sample_training_patches(Z, valid.size, 42)
    assert set(idx.tolist()) == set(valid.tolist())
    with pytest.raises(ValueError):
        sample_training_patches(Z, valid.size + 1, 42)


def test_default_config_frozen():
    c = DEFAULT_CONFIG
    assert (c["patch_size"], c["stride"], c["n_atoms"], c["n_iter"], c["train_sparsity"]) == (8, 4, 128, 8, 4)
    assert c["eval_sparsities"] == [2, 4, 8] and c["n_train_patches"] == 1500
    assert c["sigma"] == 20 / 255
    assert c["seeds"] == {"data": 42, "model": 43, "noise": 44, "sampling": 42, "ksvd": 43}
    assert c["n_train"] == c["n_train_patches"] == 1500
    assert c["constant_threshold"] == c["min_patch_norm"] == 1e-8
