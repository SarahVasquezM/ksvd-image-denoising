"""Bloque 1 - Datos y parches.

Fuente única de verdad para:
  * carga y redimensionado de las imágenes (camera -> entrenamiento, coins -> prueba);
  * extracción de parches 8x8 centrados (una columna por parche);
  * reensamblado de parches por promedio de zonas solapadas;
  * muestreo reproducible del conjunto de entrenamiento;
  * escritura de ``datos.npz``, ``config.json`` y ``versiones.txt``.
"""
from __future__ import annotations

import json
import numbers
import platform
import sys
from pathlib import Path

import numpy as np
from skimage import data as skdata
from skimage import img_as_float64
from skimage.transform import resize

CONTRACT_VERSION = 1

# Configuración experimental congelada (ver 00_general/00_Plan_general.md).
DEFAULT_CONFIG: dict = {
    "version_contrato": 1,
    "train_image_name": "camera",
    "test_image_name": "coins",
    "train_image_source": "skimage.data.camera()",
    "test_image_source": "skimage.data.coins()",
    "train_shape": [256, 256],
    "test_shape": [128, 128],
    "resize": {"anti_aliasing": True, "preserve_range": True},
    "conversion": "skimage.img_as_float64",
    "patch_size": 8,
    "stride": 4,
    "patch_order": "filas y luego columnas; flatten en orden C; un parche por columna",
    "centering": "se resta la media de cada parche (sin normalizar por la norma)",
    "min_patch_norm": 1e-8,
    "constant_threshold": 1e-8,
    "n_train_patches": 1500,
    "n_train": 1500,
    "n_atoms": 128,
    "atom_dim": 64,
    "n_iter": 8,
    "train_sparsity": 4,
    "eval_sparsities": [2, 4, 8],
    "sigma": 20 / 255,
    "sigma_255": 20,
    "data_range": 1.0,
    "seeds": {"data": 42, "model": 43, "noise": 44, "sampling": 42, "ksvd": 43},
    "images": {
        "train": {"name": "camera", "shape": [256, 256]},
        "test": {"name": "coins", "shape": [128, 128]},
    },
    "dtype": "float64",
    "flatten_order": "C",
    "position_order": "row_column",
    "center": True,
    "normalize_patch": False,
    "sample_replace": False,
}

PATCH_NORM_TOL = 1e-8


def _positive_int(value, name: str) -> int:
    """Valida enteros positivos sin aceptar booleanos como enteros."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} debe ser un entero positivo")
    return int(value)


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} debe ser un entero no negativo")
    return int(value)


def _real_array(value, name: str, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} debe ser un array numérico real de {ndim} dimensiones")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contiene NaN o infinito")
    return array


# ---------------------------------------------------------------------------
# Imágenes
# ---------------------------------------------------------------------------
def load_image(name: str, shape) -> np.ndarray:
    """Carga ``skimage.data.<name>()``, la convierte a float64 en [0, 1] y la redimensiona."""
    loader = getattr(skdata, name)
    img = img_as_float64(loader())
    if img.ndim != 2:
        raise ValueError(f"Se esperaba una imagen en escala de grises; {name} tiene forma {img.shape}")
    out = resize(img, tuple(shape), anti_aliasing=True, preserve_range=True)
    return np.ascontiguousarray(out, dtype=np.float64)


def load_images(config: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    cfg = DEFAULT_CONFIG if config is None else config
    train = load_image(cfg["train_image_name"], cfg["train_shape"])
    test = load_image(cfg["test_image_name"], cfg["test_shape"])
    return train, test


# ---------------------------------------------------------------------------
# Parches
# ---------------------------------------------------------------------------
def patch_starts(length: int, patch_size: int, stride: int) -> list[int]:
    """Inicios válidos a lo largo de un eje, incluyendo ``length - patch_size`` si el stride no llega."""
    length = _positive_int(length, "length")
    patch_size = _positive_int(patch_size, "patch_size")
    stride = _positive_int(stride, "stride")
    if patch_size > length:
        raise ValueError(f"patch_size={patch_size} mayor que la dimensión {length}")
    if stride > patch_size:
        raise ValueError("stride debe ser <= patch_size para asegurar cobertura")
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_patches(image: np.ndarray, patch_size: int = 8, stride: int = 4):
    """Extrae parches centrados.

    Returns
    -------
    Z : (patch_size**2, N) parches centrados (media del parche restada), float64.
    means : (N,) media de cada parche.
    positions : (N, 2) esquina superior izquierda ``(fila, columna)``; recorrido por filas
        y, dentro de cada fila, por columnas.
    """
    patch_size = _positive_int(patch_size, "patch_size")
    stride = _positive_int(stride, "stride")
    image = _real_array(image, "image", 2)
    H, W = image.shape
    rows = patch_starts(H, patch_size, stride)
    cols = patch_starts(W, patch_size, stride)
    positions = np.array([(r, c) for r in rows for c in cols], dtype=np.int64)
    n = positions.shape[0]
    P = np.empty((patch_size * patch_size, n), dtype=np.float64)
    for k, (r, c) in enumerate(positions):
        P[:, k] = image[r:r + patch_size, c:c + patch_size].flatten(order="C")
    means = P.mean(axis=0)
    Z = P - means[None, :]
    return Z, means, positions


def assemble_patches(patches: np.ndarray, positions: np.ndarray, image_shape, patch_size: int = 8,
                     return_coverage: bool = False):
    """Reconstruye una imagen promediando parches (YA con la media restaurada) en zonas solapadas.

    No aplica clipping. Lanza ``ValueError`` si algún píxel no queda cubierto.
    """
    patch_size = _positive_int(patch_size, "patch_size")
    patches = _real_array(patches, "patches", 2)
    positions = np.asarray(positions)
    try:
        shape = tuple(image_shape)
    except TypeError as exc:
        raise ValueError("image_shape debe contener alto y ancho") from exc
    if len(shape) != 2:
        raise ValueError("image_shape debe contener alto y ancho")
    H, W = (_positive_int(v, "image_shape") for v in shape)
    if min(H, W) < patch_size:
        raise ValueError("image_shape debe ser >= patch_size")
    if patches.shape != (patch_size * patch_size, positions.shape[0]):
        raise ValueError(f"patches tiene forma {patches.shape}; se esperaba "
                         f"({patch_size * patch_size}, {positions.shape[0]})")
    if positions.ndim != 2 or positions.shape[1:] != (2,) or positions.dtype.kind not in "iu":
        raise ValueError("positions debe tener forma (N, 2) y tipo entero")
    if (np.any(positions < 0) or np.any(positions[:, 0] > H - patch_size)
            or np.any(positions[:, 1] > W - patch_size)):
        raise ValueError("Hay posiciones fuera de la imagen")
    acc = np.zeros((H, W), dtype=np.float64)
    cov = np.zeros((H, W), dtype=np.int64)
    for k, (r, c) in enumerate(positions):
        acc[r:r + patch_size, c:c + patch_size] += patches[:, k].reshape(patch_size, patch_size)
        cov[r:r + patch_size, c:c + patch_size] += 1
    if np.any(cov == 0):
        raise ValueError(f"Mapa de cobertura con {int(np.sum(cov == 0))} píxeles en cero")
    out = acc / cov
    if not np.all(np.isfinite(out)):
        raise ValueError("El ensamblado produjo valores no finitos")
    return (out, cov) if return_coverage else out


def roundtrip_error(image: np.ndarray, patch_size: int = 8, stride: int = 4) -> float:
    """Error máximo absoluto de extraer + reensamblar (prueba obligatoria del contrato)."""
    Z, means, positions = extract_patches(image, patch_size, stride)
    rec = assemble_patches(Z + means[None, :], positions, image.shape, patch_size)
    return float(np.max(np.abs(rec - image)))


def sample_training_patches(Z: np.ndarray, n_samples: int, seed: int, min_norm: float = PATCH_NORM_TOL):
    """Filtra parches con norma centrada <= ``min_norm`` y elige ``n_samples`` sin reemplazo.

    Devuelve ``(X_train, train_indices)``; los índices se refieren a las columnas de la
    extracción COMPLETA (antes del filtrado).
    """
    Z = _real_array(Z, "Z", 2)
    n_samples = _positive_int(n_samples, "n_samples")
    seed = _nonnegative_int(seed, "seed")
    if not np.isfinite(min_norm) or min_norm < 0:
        raise ValueError("min_norm debe ser finito y no negativo")
    norms = np.linalg.norm(Z, axis=0)
    valid = np.flatnonzero(norms > min_norm)
    if valid.size < n_samples:
        raise ValueError(f"Sólo hay {valid.size} parches válidos; se piden {n_samples}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(valid, size=n_samples, replace=False).astype(np.int64)
    return np.ascontiguousarray(Z[:, idx]), idx


# ---------------------------------------------------------------------------
# Productos del bloque
# ---------------------------------------------------------------------------
def library_versions() -> dict:
    import matplotlib
    import scipy
    import skimage
    import sklearn
    vers = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "scikit-image": skimage.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import pandas
        vers["pandas"] = pandas.__version__
    except ImportError:  # pragma: no cover
        pass
    return vers


def build_dataset(output_dir, config: dict | None = None) -> dict:
    """Genera ``datos.npz``, ``config.json`` y ``versiones.txt`` en ``output_dir``.

    Devuelve un diccionario con resúmenes y comprobaciones.
    """
    cfg = dict(DEFAULT_CONFIG if config is None else config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train, test = load_images(cfg)
    p, s = cfg["patch_size"], cfg["stride"]

    checks = {
        "roundtrip_max_error_train": roundtrip_error(train, p, s),
        "roundtrip_max_error_test": roundtrip_error(test, p, s),
    }
    for k, v in checks.items():
        if not v < 1e-10:
            raise AssertionError(f"{k}={v} >= 1e-10")

    Z, means, positions = extract_patches(train, p, s)
    norms = np.linalg.norm(Z, axis=0)
    X_train, train_indices = sample_training_patches(
        Z, cfg["n_train_patches"], cfg["seeds"]["sampling"], cfg["min_patch_norm"])
    if not np.array_equal(X_train, Z[:, train_indices]):
        raise AssertionError("X_train no coincide con Z[:, train_indices]")

    Zt, _, post = extract_patches(test, p, s)
    summary = {
        "n_patches_train_total": int(Z.shape[1]),
        "n_patches_train_valid": int(np.sum(norms > cfg["min_patch_norm"])),
        "n_patches_train_discarded": int(np.sum(norms <= cfg["min_patch_norm"])),
        "n_patches_test_total": int(Zt.shape[1]),
        "train_range": [float(train.min()), float(train.max())],
        "test_range": [float(test.min()), float(test.max())],
        **checks,
    }

    np.savez(out / "datos.npz",
             train_image=train.astype(np.float64),
             test_image=test.astype(np.float64),
             X_train=X_train.astype(np.float64),
             train_indices=train_indices.astype(np.int64))

    cfg_out = dict(cfg)
    cfg_out["dimensions"] = {
        "train_image": list(train.shape),
        "test_image": list(test.shape),
        "X_train": list(X_train.shape),
        "train_indices": list(train_indices.shape),
        "n_patches_train_total": summary["n_patches_train_total"],
        "n_patches_test_total": summary["n_patches_test_total"],
    }
    cfg_out["checks"] = checks
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2, ensure_ascii=False)

    vers = library_versions()
    with open(out / "versiones.txt", "w", encoding="utf-8") as f:
        for k, v in vers.items():
            f.write(f"{k}: {v}\n")

    return {"summary": summary, "config": cfg_out, "versions": vers}


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset(path) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Genera los productos del bloque de datos")
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "01_datos"))
    args = ap.parse_args()
    res = build_dataset(args.output_dir)
    print(json.dumps(res["summary"], indent=2))
