# Nota del bloque 1 — Datos y parches

**Responsable original:** integrante 1. **Implementación:** `src/preprocesamiento.py` (fuente única;
`01_datos/preprocesamiento.py` sólo la reexporta). **Notebook:** `01_datos.ipynb`.

## Objetivo
Entregar al bloque 2 una matriz de entrenamiento `X_train` (64 × 1500) de parches centrados y al
bloque 3 la imagen de prueba, junto con un `config.json` que fija el contrato del experimento
(tamaños, semillas, parámetros). Ningún bloque posterior vuelve a leer `skimage.data`: todos leen
`datos.npz` y `config.json`.

## Procedencia
| Uso | Fuente | Original | Tras redimensionar |
|---|---|---|---|
| Entrenamiento | `skimage.data.camera()` | 512 × 512, uint8 | 256 × 256 |
| Prueba | `skimage.data.coins()` | 303 × 384, uint8 | 128 × 128 |

Ambas imágenes se distribuyen con scikit-image (no requieren descarga) y son de escala de grises.

## Preparación
1. `img_as_float64` → valores en [0, 1].
2. `skimage.transform.resize(..., anti_aliasing=True, preserve_range=True)` al tamaño de la tabla.
   *coins* no es cuadrada: el redimensionado a 128 × 128 cambia su relación de aspecto (decisión
   congelada de la configuración, documentada como limitación).

## Separación entrenamiento / prueba
El diccionario sólo ve parches de *camera*. *coins* se usa exclusivamente en la evaluación: no hay
fuga de información entre imágenes.

## Parches
* Tamaño 8 × 8, paso (`stride`) 4. Las esquinas superiores izquierdas se recorren **por filas y,
  dentro de cada fila, por columnas**. Si el paso no llega exactamente al borde se añade el último
  inicio válido `H − 8` / `W − 8`, sin duplicar posiciones (en 256 y 128 el paso sí llega: 248 y 120).
* Cada parche se aplana en orden C y se guarda como **columna**.
* Resultado: camera → 63 × 63 = **3969** parches; coins → 31 × 31 = **961** parches.
* `extract_patches(image, patch_size=8, stride=4)` → `Z (64, N)`, `means (N,)`, `positions (N, 2)`.
* `assemble_patches(patches, positions, image_shape, patch_size=8)` recibe parches **con la media
  restaurada**, promedia zonas solapadas, no hace *clipping* y falla si el mapa de cobertura tiene ceros.

## Centrado
A cada parche se le resta su media (guardada en `means`). **No** se divide entre su norma: la
amplitud de los bordes se conserva y la evaluación restaura la media antes de reensamblar.

## Muestreo
1. Se descartan parches con norma centrada ≤ 1e-8 (parches constantes). En camera 256 × 256 no se
   descartó ninguno (3969 válidos; la norma mínima de los muestreados es ≈ 7.2e-3).
2. Se eligen **1500 sin reemplazo** con `np.random.default_rng(42).choice(validos, 1500, replace=False)`.
3. `train_indices` (1500,) son índices de columna de la extracción **completa** previa al filtrado,
   en el orden en que los devuelve el RNG (no ordenados). Se cumple `X_train == Z[:, train_indices]`.

## Semillas
| Uso | Semilla |
|---|---|
| Muestreo de parches | 42 |
| Inicialización K-SVD | 43 |
| Ruido de evaluación | 44 |

## Productos
| Archivo | Contenido |
|---|---|
| `datos.npz` | `train_image` (256,256) f64, `test_image` (128,128) f64, `X_train` (64,1500) f64, `train_indices` (1500,) int64 |
| `config.json` | contrato v1.0: tamaños, parámetros, semillas, nombres de imágenes, dimensiones y comprobaciones |
| `versiones.txt` | versiones de Python y bibliotecas con las que se generaron los datos |
| `figuras/imagenes_camera_coins.png` | imágenes de entrenamiento y prueba |
| `figuras/muestra_parches.png` | ubicación y aspecto de 64 parches muestreados |

## Comprobaciones (ejecutadas, no supuestas)
* Ida y vuelta obligatoria `assemble_patches(Z + means, positions, shape)`: error máximo
  **1.1e-16** en camera y en coins (< 1e-10). Se ejecuta en `build_dataset`, en el notebook y en
  `tests/test_preprocesamiento.py`.
* Orden de recorrido, inclusión del último inicio, ausencia de duplicados, no-clipping y error ante
  cobertura incompleta: pruebas unitarias.
* Reproducibilidad: dos ejecuciones producen `datos.npz` idéntico bit a bit (`tests/test_integracion.py`).

## Limitaciones
* Una sola imagen de entrenamiento (≈ 1500 de 3969 parches) — el diccionario es pequeño y específico.
* coins se deforma al pasar a 128 × 128.
* El *anti-aliasing* del redimensionado suaviza las imágenes; los resultados no son comparables con
  experimentos sobre imágenes a resolución completa.
