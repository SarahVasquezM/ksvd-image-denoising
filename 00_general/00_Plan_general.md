# Plan general

## Objetivo científico
Adaptación didáctica pequeña de K-SVD para:
1. aprender un diccionario disperso a partir de parches de una imagen (camera);
2. reconstruir una imagen limpia no usada en entrenamiento (coins);
3. reconstruir una versión ruidosa de esa imagen (σ = 20/255);
4. estudiar el efecto de la dispersión T0 ∈ {2, 4, 8};
5. evaluar con MSE, PSNR, actividad dispersa y tiempo de reconstrucción.

No es una réplica de los artículos originales y no pretende superar el estado del arte.

## Reparto original en cuatro bloques
El trabajo se diseñó para cuatro integrantes. En esta entrega un solo desarrollador lo implementó
completo, pero se conserva la división para que cada integrante pueda revisar su bloque por separado.

| Carpeta | Bloque | Entradas | Productos |
|---|---|---|---|
| `01_datos/` | Datos y parches | `skimage.data` | `datos.npz`, `config.json`, `versiones.txt`, `nota_datos.md` |
| `02_modelo/` | OMP + K-SVD | `datos.npz`, `config.json` | `modelo.npz`, `historial.csv`, `config_efectiva.json`, `nota_metodo.md` |
| `03_evaluacion/` | Experimentos | datos + modelo + config | `resultados.csv`, `reconstrucciones.npz`, `nota_resultados.md` |
| `04_final/` | Integración | todo lo anterior | figuras finales, reporte, presentación, `README_integracion.md` |

Los bloques se comunican **sólo por archivos** (contrato v1.0 en `config.json`). La implementación
es **única** y vive en `src/`; las carpetas contienen wrappers, notebooks, notas y productos.

## Configuración experimental congelada
| Parámetro | Valor |
|---|---|
| Entrenamiento / prueba | `camera` 256×256 / `coins` 128×128 (`img_as_float64`, `resize` con `anti_aliasing=True`, `preserve_range=True`) |
| Parches | 8×8, stride 4, filas→columnas, flatten C, centrados (sin normalizar) |
| Muestreo | filtrar norma ≤ 1e-8, 1500 sin reemplazo, RNG 42 |
| Diccionario | K = 128, átomos de dimensión 64, columnas normalizadas |
| Entrenamiento | 8 iteraciones, T = 4, semilla 43 |
| Evaluación | T0 ∈ {2, 4, 8}; ruido gaussiano σ = 20/255, semilla 44, una realización |
| Métricas | MSE, PSNR (`data_range=1.0`), coeficientes activos medios, tiempo |

## Requisitos técnicos clave
* OMP: `sklearn.linear_model.orthogonal_mp` (no reimplementado).
* SVD: `numpy.linalg.svd` sobre el residual restringido (no reimplementada).
* Ciclo K-SVD propio; no se usa `DictionaryLearning`.
* Contingencia de tiempo sólo tras medir (ver `incidencias.md`).

## Estado
Completado: los cuatro bloques se ejecutaron, los productos están versionados, 31+ pruebas pasan y
la auditoría desde un entorno limpio se documenta en `reproduccion.md`.
