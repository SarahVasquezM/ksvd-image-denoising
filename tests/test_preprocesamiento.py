"""Pruebas pequeñas del contrato de integración; ejecutar con unittest."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from preprocesamiento import extract_patches, assemble_patches, sample_training, prepare_data, validate_data


class PatchTests(unittest.TestCase):
    def test_order_and_single_patch(self):
        im = np.arange(64).reshape(8, 8)
        Z, m, pos = extract_patches(im)
        np.testing.assert_array_equal(Z[:, 0] + m[0], im.ravel(order='C'))
        self.assertEqual(Z.shape, (64, 1))
        self.assertEqual(Z.dtype, np.float64)
        np.testing.assert_array_equal(pos, [[0, 0]])

    def test_irregular_edges_and_no_clipping(self):
        im = np.arange(11*17, dtype=np.float64).reshape(11, 17) / 10 - 2
        Z, m, pos = extract_patches(im)
        expected = np.array([(r, c) for r in (0, 3) for c in (0, 4, 8, 9)])
        np.testing.assert_array_equal(pos, expected)
        self.assertEqual(len(np.unique(pos, axis=0)), len(pos))
        actual = assemble_patches(Z+m[None, :], pos, im.shape)
        self.assertLess(np.max(np.abs(actual-im)), 1e-10)
        self.assertLess(actual.min(), 0)
        self.assertGreater(actual.max(), 1)

    def test_constant(self):
        im = np.full((13, 15), 0.3)
        Z, m, pos = extract_patches(im)
        np.testing.assert_allclose(Z, 0, atol=1e-15)
        np.testing.assert_allclose(assemble_patches(Z+m, pos, im.shape), im, atol=1e-15)
        with self.assertRaisesRegex(ValueError, 'elegibles'):
            sample_training(Z, n_train=1)

    def test_sample_indices_before_filter(self):
        Z = np.array([[0, 1, 0, 2, 3], [0, -1, 0, -2, -3]], dtype=float)
        X, idx = sample_training(Z, n_train=3)
        self.assertEqual(set(idx), {1, 3, 4})
        np.testing.assert_array_equal(X, Z[:, idx])
        np.testing.assert_array_equal(idx, sample_training(Z, n_train=3)[1])

    def test_overlap_average(self):
        patches = np.column_stack([np.ones(4), np.full(4, 3.)])
        result = assemble_patches(patches, np.array([[0, 0], [0, 1]]), (2, 3), 2)
        np.testing.assert_array_equal(result, [[1, 2, 3], [1, 2, 3]])

    def test_invalid_inputs(self):
        for image, p, s in [(np.ones((7, 8)), 8, 4), (np.ones((8, 8, 1)), 8, 4),
                            (np.full((8, 8), np.nan), 8, 4), (np.ones((8, 8)), 0, 4),
                            (np.ones((8, 8)), 8, 0), (np.ones((8, 8)), 8, 9),
                            (np.ones((8, 8)), 8.0, 4), (np.ones((8, 8)), True, 4)]:
            with self.subTest(p=p, s=s, shape=image.shape), self.assertRaises(ValueError):
                extract_patches(image, p, s)
        for patches, pos, shape in [(np.ones((64, 1)), [[0., 0.]], (8, 8)),
                                    (np.ones((64, 1)), [[-1, 0]], (8, 8)),
                                    (np.ones((64, 1)), [[0, 0]], (9, 9)),
                                    (np.ones((63, 1)), [[0, 0]], (8, 8)),
                                    (np.full((64, 1), np.inf), [[0, 0]], (8, 8))]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                assemble_patches(patches, pos, shape)

    def test_real_data_contract(self):
        arrays, _ = prepare_data()
        evidence = validate_data(arrays)
        self.assertEqual(evidence['train_image']['patches'], 3969)
        self.assertEqual(evidence['test_image']['patches'], 961)


if __name__ == '__main__':
    unittest.main()
