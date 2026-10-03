"""Datos y parches del contrato 1. No contiene entrenamiento ni denoising."""
from pathlib import Path
import json
import numbers
import numpy as np


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} debe ser un entero positivo")
    return int(value)


def _real_array(value, name, ndim):
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in 'iuf':
        raise ValueError(f"{name} debe ser un array numérico real de {ndim} dimensiones")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contiene NaN o infinito")
    return array


def _starts(length, patch_size, stride):
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_patches(image, patch_size=8, stride=4):
    """Devuelve Z=(p*p,N), medias=(N,) y posiciones=(N,2), orden C.

    El paso no puede superar el parche: así no se dejan huecos interiores.
    Se añade el último inicio válido por eje si el paso no llega al borde.
    No normaliza intensidades ni normas; convierte la representación a float64.
    """
    p = _positive_int(patch_size, 'patch_size')
    s = _positive_int(stride, 'stride')
    image = _real_array(image, 'image', 2)
    if min(image.shape) < p:
        raise ValueError('image debe tener ambos lados >= patch_size')
    if s > p:
        raise ValueError('stride debe ser <= patch_size para asegurar cobertura')
    positions = np.array([(r, c) for r in _starts(image.shape[0], p, s)
                          for c in _starts(image.shape[1], p, s)], dtype=np.int64)
    patches = np.column_stack([image[r:r+p, c:c+p].ravel(order='C')
                               for r, c in positions])
    means = patches.mean(axis=0)
    return patches - means[None, :], means, positions


def assemble_patches(patches, positions, image_shape, patch_size=8):
    """Promedia solapamientos de columnas con la media YA restituida. Sin clipping."""
    p = _positive_int(patch_size, 'patch_size')
    try:
        shape = tuple(image_shape)
    except TypeError as exc:
        raise ValueError('image_shape debe contener alto y ancho') from exc
    if len(shape) != 2:
        raise ValueError('image_shape debe contener alto y ancho')
    h, w = (_positive_int(v, 'image_shape') for v in shape)
    if min(h, w) < p:
        raise ValueError('image_shape debe ser >= patch_size')
    patches = _real_array(patches, 'patches', 2)
    positions = np.asarray(positions)
    if (patches.shape[0] != p*p or positions.shape != (patches.shape[1], 2)
            or positions.dtype.kind not in 'iu'):
        raise ValueError('Se esperan patches=(p*p,N) y positions enteras=(N,2)')
    if (np.any(positions < 0) or np.any(positions[:, 0] > h-p)
            or np.any(positions[:, 1] > w-p)):
        raise ValueError('Hay posiciones fuera de la imagen')
    total = np.zeros((h, w), dtype=np.float64)
    count = np.zeros((h, w), dtype=np.int64)
    for column, (r, c) in enumerate(positions):
        total[r:r+p, c:c+p] += patches[:, column].reshape(p, p, order='C')
        count[r:r+p, c:c+p] += 1
    if np.any(count == 0):
        raise ValueError('Hay píxeles sin cobertura')
    result = total / count
    if not np.isfinite(result).all():
        raise ValueError('El ensamblado desbordó float64')
    return result


def base_config():
    """Parámetros congelados; los de otros bloques sólo se transmiten."""
    return dict(version_contrato=1, patch_size=8, stride=4, n_train=1500,
                constant_threshold=1e-8, n_atoms=128, n_iter=8, train_sparsity=4,
                eval_sparsities=[2, 4, 8], sigma=20/255,
                seeds=dict(data=42, model=43, noise=44),
                images=dict(train=dict(name='camera', shape=[256, 256]),
                            test=dict(name='coins', shape=[128, 128])),
                dtype='float64', flatten_order='C', position_order='row_column',
                center=True, normalize_patch=False, sample_replace=False,
                rng='numpy.random.default_rng (PCG64)',
                resize=dict(anti_aliasing=True, preserve_range=True, order=1,
                            mode='reflect', clip=True, anti_aliasing_sigma=None),
                conversion='img_as_float64 before resize')


def sample_training(Z, n_train=1500, seed=42, threshold=1e-8):
    """Índices referidos a Z completo, nunca a la matriz filtrada."""
    Z = _real_array(Z, 'Z', 2)
    n_train = _positive_int(n_train, 'n_train')
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold debe ser finito y no negativo')
    eligible = np.flatnonzero(np.linalg.norm(Z, axis=0) > threshold)
    if eligible.size < n_train:
        raise ValueError(f'Sólo hay {eligible.size} parches elegibles para {n_train}')
    indices = np.random.default_rng(seed).choice(eligible, size=n_train, replace=False)
    return Z[:, indices].copy(), indices.astype(np.int64)


def prepare_data():
    """Prepara camera/coins y la muestra sin utilizar la prueba para aprender."""
    from skimage import data, img_as_float64
    from skimage.transform import resize
    config = base_config()
    train = resize(img_as_float64(data.camera()), (256, 256), **config['resize'])
    test = resize(img_as_float64(data.coins()), (128, 128), **config['resize'])
    Z, _, _ = extract_patches(train)
    X, indices = sample_training(Z)
    return dict(train_image=train, test_image=test, X_train=X, train_indices=indices), config


def validate_data(arrays):
    """Verifica el contrato y devuelve evidencia numérica serializable."""
    expected = dict(train_image=(256, 256), test_image=(128, 128),
                    X_train=(64, 1500), train_indices=(1500,))
    if set(arrays) != set(expected):
        raise ValueError('Claves del paquete diferentes del contrato')
    for key, shape in expected.items():
        a = arrays[key]
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'Forma/finitud inválida: {key}')
        if key == 'train_indices':
            if a.dtype.kind not in 'iu':
                raise ValueError('train_indices debe ser entero')
        elif a.dtype != np.float64:
            raise ValueError(f'{key} debe ser float64')
    evidence = {}
    for key in ('train_image', 'test_image'):
        im = arrays[key]
        if im.min() < 0 or im.max() > 1:
            raise ValueError('Imágenes fuera de [0,1]')
        Z, means, positions = extract_patches(im)
        error = float(np.max(np.abs(assemble_patches(Z + means[None, :], positions, im.shape) - im)))
        if error >= 1e-10:
            raise ValueError('Reconstrucción fuera de tolerancia')
        count = np.zeros(im.shape, dtype=np.int64)
        for r, c in positions:
            count[r:r+8, c:c+8] += 1
        if count.min() <= 0:
            raise ValueError('Cobertura incompleta')
        evidence[key] = dict(shape=list(im.shape), range=[float(im.min()), float(im.max())],
                             patches=Z.shape[1], max_error=error,
                             coverage_min=int(count.min()), coverage_max=int(count.max()))
    Z, _, _ = extract_patches(arrays['train_image'])
    X, indices = sample_training(Z)
    if not np.array_equal(indices, arrays['train_indices']) or not np.array_equal(X, arrays['X_train']):
        raise ValueError('Muestra/índices no corresponden a extracción y semilla 42')
    if np.unique(indices).size != 1500 or not np.array_equal(Z[:, indices], X):
        raise ValueError('Índices repetidos o incorrectos')
    mean_error = float(np.max(np.abs(X.mean(axis=0))))
    min_norm = float(np.linalg.norm(X, axis=0).min())
    if mean_error >= 1e-12 or min_norm <= 1e-8:
        raise ValueError('Centrado o filtro de constantes incorrecto')
    evidence.update(centered_mean_max=mean_error, selected_min_norm=min_norm,
                    eligible_patches=int(np.sum(np.linalg.norm(Z, axis=0) > 1e-8)),
                    reproducible=True, original_indices_verified=True)
    return evidence


def save_data(output_dir):
    """Genera arrays, configuración, versiones, figuras y comprobación de recarga."""
    import importlib.metadata
    import platform
    import sklearn  # Verificación de disponibilidad para integrante 2.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays, config = prepare_data()
    evidence = validate_data(arrays)
    np.savez_compressed(output / 'datos.npz', **arrays)
    with np.load(output / 'datos.npz', allow_pickle=False) as loaded:
        validate_data(dict(loaded))
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, loaded[key])
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    if json.loads((output / 'config.json').read_text()) != config:
        raise ValueError('Configuración no recuperada correctamente')
    versions = [f'Python=={platform.python_version()}']
    versions += [f'{p}=={importlib.metadata.version(p)}' for p in
                 ('numpy', 'scipy', 'scikit-learn', 'scikit-image', 'matplotlib')]
    (output / 'versiones.txt').write_text('\n'.join(versions) + '\n', encoding='utf-8')
    evidence['saved_arrays_verified'] = True
    evidence['execution_environment'] = 'Kaggle' if Path('/kaggle/working').is_dir() else 'local'
    (output / 'comprobaciones.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
    for ax, key, title in zip(axes, ('train_image', 'test_image'),
                              ('Entrenamiento: camera (256 x 256)', 'Prueba: coins (128 x 128)')):
        ax.imshow(arrays[key], cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    fig.savefig(output / 'imagenes.png', dpi=160)
    plt.close(fig)
    Z, means, _ = extract_patches(arrays['train_image'])
    fig, axes = plt.subplots(4, 4, figsize=(6, 6), layout='constrained')
    for ax, idx in zip(axes.flat, arrays['train_indices'][:16]):
        ax.imshow((Z[:, idx] + means[idx]).reshape(8, 8), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Índice {idx}', fontsize=9)
        ax.axis('off')
    fig.suptitle('Primeros 16 parches seleccionados; media restituida')
    fig.savefig(output / 'parches.png', dpi=160)
    plt.close(fig)
    write_note(output, evidence)
    return evidence


def write_note(output, evidence):
    """Nota factual de cada ejecución, sin resultados del resto del equipo."""
    text = f'''# Nota de datos: Dani, paquete 1

## Texto para el reporte

El objetivo del proyecto es estudiar una representación dispersa por parches
para reconstrucción y eliminación de ruido con un diccionario aprendido mediante
K-SVD. Este bloque prepara los datos y verifica las operaciones de extracción y
ensamblado necesarias para las etapas posteriores; no ejecuta entrenamiento ni
evalúa eliminación de ruido. Es una adaptación didáctica del enfoque por parches
de Elad y Aharon (2006, DOI: 10.1109/TIP.2006.881969), no una reproducción de su
estimador bayesiano completo ni de sus resultados.

Se utilizaron dos imágenes de ejemplo distribuidas por scikit-image: camera
exclusivamente para entrenamiento y coins exclusivamente para prueba. Ambas se
convirtieron mediante img_as_float64 antes de redimensionarlas, respectivamente,
a 256×256 y 128×128, con anti_aliasing=True y preserve_range=True. Se conservaron
arrays float64 en escala de grises y rango [0,1]. La procedencia y las condiciones
de las imágenes se consultan en la documentación de skimage.data; no se asigna
una licencia nueva a las imágenes ni al proyecto.

Se extrajeron ventanas de 8×8 con paso 4, recorriendo filas y después columnas.
Cada ventana se aplanó en orden C para formar una columna de longitud 64. Se
restó su media, conservándola para invertir el centrado, sin normalizar la norma
del parche. De los 3969 parches completos de entrenamiento, se conservaron como
elegibles los de norma centrada mayor que 1e-8 y se seleccionaron 1500 sin
reemplazo mediante default_rng(42). Los índices guardados corresponden a la
extracción completa anterior al filtrado. La imagen de prueba produjo 961 parches.

El ensamblado suma las contribuciones y divide por su cobertura, sin clipping.
Se verificaron formas, tipos, finitud, centrado, reproducibilidad, índices y carga
de los archivos. Restituir la media y ensamblar recuperó ambas imágenes con error
máximo menor que 1e-10. El paquete transmite las semillas 43 y 44 para modelo y
ruido, sin utilizarlas aquí. Dos imágenes y una sola imagen de prueba no permiten
afirmar generalización; no existe conjunto de validación ni resultados de denoising
en esta entrega.

## Evidencia de esta ejecución

- Entorno: {evidence['execution_environment']} (sólo este entorno está verificado por esta ejecución).
- Error máximo camera: {evidence['train_image']['max_error']:.17g}.
- Error máximo coins: {evidence['test_image']['max_error']:.17g}.
- Mayor media absoluta de columnas: {evidence['centered_mean_max']:.17g}.
- Norma mínima seleccionada: {evidence['selected_min_norm']:.17g}.
- Parches elegibles: {evidence['eligible_patches']}.
- Cobertura mínima: 1 en ambas imágenes; valores y rangos en comprobaciones.json.
- Recarga NPZ sin pickle y configuración JSON verificadas.

## Reproducción y entrega

Ejecutar 01_datos.ipynb completo en una sesión nueva. Importa preprocesamiento.py
del repositorio o del recurso privado montado. Guardar una versión con salidas;
descargar entrega_01 y conservar su manifiesto SHA-256 y versiones.txt. Los
parámetros de otros integrantes están en config.json, pero no prueban ejecución.

## Explicación oral breve

Mi parte convierte dos imágenes en señales pequeñas que el siguiente integrante
podrá usar. Un parche de 8 por 8 se vuelve una columna de 64 números. Le resto
su promedio para separar el brillo local de las variaciones y guardo ese promedio.
Selecciono 1500 parches de camera con una semilla fija; coins queda reservada para
probar. Al reconstruir, devuelvo cada media una sola vez y promedio donde los
parches se superponen. Verifiqué que esta ida y vuelta recupera las imágenes
prácticamente sin error. Eso comprueba mi preparación, no la calidad del denoising.

Preguntas clave: ¿por qué parches? Reducen la dimensión y capturan estructura local.
¿Por qué solapar? Permite promediar varias estimaciones por píxel. ¿Por qué guardar
la media? Sin ella perderíamos el brillo. ¿Por qué separar camera y coins? Para
evitar aprender con la misma imagen limpia que se evaluará.
'''
    (Path(output) / 'nota_datos.md').write_text(text, encoding='utf-8')
