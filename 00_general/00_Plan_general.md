# Plan general

## Objetivo científico
Adaptación didáctica pequeña de K-SVD para:
1. aprender un diccionario externo con parches de `camera` y uno adaptativo con parches de la
   observación ruidosa de `coins`;
2. reconstruir una imagen limpia no usada en entrenamiento (coins);
3. reconstruir una versión ruidosa de esa imagen (σ = 20/255);
4. estudiar el efecto de la dispersión T0 ∈ {2, 4, 8};
5. evaluar con MSE, PSNR, actividad dispersa y tiempo de reconstrucción.

No es una réplica de los artículos originales y no pretende superar el estado del arte.

## Reparto original en cuatro bloques
El trabajo se divide en cuatro integrantes: Sarah prepara datos y parches; Alan (Edgar) implementa
OMP, K-SVD y entrenamiento; Sergio realiza experimentos y métricas; Areli integra reporte y presentación.

| Carpeta | Bloque | Entradas | Productos |
|---|---|---|---|
| `01_datos/` | Datos y parches | `skimage.data` | `datos.npz`, `config.json`, `versiones.txt`, `nota_datos.md` |
| `02_modelo/` | OMP + K-SVD | `datos.npz`, `config.json` | `modelo.npz`, `historial.csv`, `config_efectiva.json`, `nota_metodo.md` |
| `03_evaluacion/` | Experimentos | datos + modelo + config | `resultados.csv`, `reconstrucciones.npz`, `nota_resultados.md` |
| `04_final/` | Integración | todo lo anterior | figuras finales, artículo IEEE y presentación editable |
| `05_comparacion/` | Extensión adaptativa | datos + modelo externo + config | modelo adaptativo, comparación, validaciones y figuras |

Los bloques se comunican **sólo por archivos** (contrato v1.0 en `config.json`). La implementación
es **única** y vive en `src/`; las carpetas contienen wrappers, notebooks, notas y productos.

## Configuración experimental congelada
| Parámetro | Valor |
|---|---|
| Entrenamiento / prueba | ruta externa: `camera` 256×256; ruta adaptativa: `coins` ruidosa 128×128; referencia: `coins` limpia |
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
Completado: los cuatro bloques originales y la extensión adaptativa se ejecutaron, los productos
están versionados, 35 pruebas pasan y la validación se documenta en `reproduccion.md`.
