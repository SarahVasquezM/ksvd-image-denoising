# K-SVD para reconstrucción y eliminación de ruido en imágenes

Adaptación **didáctica y pequeña** de K-SVD (Aharon, Elad y Bruckstein, 2006) y de su uso para
eliminar ruido (Elad y Aharon, 2006): se aprende un diccionario de 128 átomos a partir de parches
8×8 de `camera` y se reconstruye `coins` (no vista en entrenamiento), limpia y con ruido gaussiano
σ = 20/255, para T0 = 2, 4 y 8.

> No es una réplica de los artículos originales ni pretende superar el estado del arte.

> **Procedencia.** Este es el mismo proyecto desarrollado en el repositorio de Alan (Edgar),
> integrado en SarahVasquezM/ksvd-image-denoising. Se conservaron los datos y resultados
> experimentales; únicamente se incorporaron validaciones de entrada, compatibilidad de contratos
> y los entregables finales solicitados.

## Resultados (ejecutados; `03_evaluacion/resultados.csv`)

| entrada | T0 | MSE | PSNR (dB) | coef. activos |
|---|---:|---:|---:|---:|
| ruidosa sin procesar | — | 0.006179 | 22.09 | — |
| limpia | 2 / 4 / 8 | 0.002132 / 0.001264 / 0.000719 | 26.71 / 28.98 / 31.43 | 2 / 4 / 8 |
| ruidosa | 2 / 4 / 8 | 0.002784 / 0.002442 / 0.002730 | 25.55 / **26.12** / 25.64 | 2 / 4 / 8 |

* Entrenamiento K-SVD (K = 128, 1500 parches, 8 iteraciones, T = 4): 2.09 s en la última ejecución local, **sin
  contingencia**; MSE de entrenamiento 9.55e-4 (`D_init`) → 3.41e-4.
* Imagen ruidosa: mejor T0 = 4, +4.03 dB sobre la entrada. Imagen limpia: el PSNR crece con T0.

![resumen](04_final/figuras/resumen_resultados.png)

## Estructura

```
├── 00_general/      plan, arquitectura, metodología, reproducción, incidencias
├── 01_datos/        bloque 1 (integrante 1): datos y parches     → datos.npz, config.json
├── 02_modelo/       bloque 2 (integrante 2): OMP + K-SVD         → modelo.npz, historial.csv
├── 03_evaluacion/   bloque 3 (integrante 3): experimentos        → resultados.csv, reconstrucciones.npz
├── 04_final/        bloque 4 (integrante 4): integración, reporte y presentación
├── src/             ÚNICA implementación (preprocesamiento, modelo, evaluacion, visualizacion, pipeline)
├── tests/           pruebas unitarias y de integración (pytest)
├── tools/           generación de notebooks, reporte y presentación
└── run_all.py       ejecuta los cuatro bloques
```

Cada carpeta de bloque contiene su notebook ejecutado, un wrapper `.py` que reexporta desde `src/`,
su nota técnica, sus productos y sus figuras. Guía de revisión por integrante:

| Integrante | Revisar | Documento principal |
|---|---|---|
| 1 | `01_datos/`, `src/preprocesamiento.py`, `tests/test_preprocesamiento.py` | `01_datos/nota_datos.md` |
| 2 | `02_modelo/`, `src/modelo.py`, `tests/test_modelo.py` | `02_modelo/nota_metodo.md` |
| 3 | `03_evaluacion/`, `src/evaluacion.py`, `tests/test_evaluacion.py` | `03_evaluacion/nota_resultados.md` |
| 4 | `04_final/`, `src/pipeline.py`, `tools/`, `tests/test_integracion.py` | `04_final/reporte/reporte.tex` |

## Requisitos clave de implementación
* OMP con `sklearn.linear_model.orthogonal_mp` (no reimplementado).
* SVD con `numpy.linalg.svd` sobre el **residual restringido** a las señales que usan cada átomo.
* Ciclo K-SVD propio (`src/modelo.py::fit_ksvd`); **no** se usa `DictionaryLearning`.

## Uso rápido

```bash
python -m venv .venv && source .venv/bin/activate     # o: conda env create -f environment.yml
pip install -r requirements.txt
python run_all.py                  # regenera los productos de los 4 bloques (≈ 10 s)
python tools/build_report.py       # compila el reporte si hay un motor LaTeX disponible
python -m pytest                   # 32 pruebas, incluye reproducción completa del pipeline
bash tools/execute_notebooks.sh    # regenera y ejecuta los 4 notebooks
```

Kaggle (CPU): ver `00_general/reproduccion.md`. Los notebooks detectan Kaggle y clonan el repositorio.
El [dataset privado v1](https://www.kaggle.com/datasets/sarahvasquez97/ksvd-dani-datos-v1)
y el [notebook privado ejecutado](https://www.kaggle.com/code/sarahvasquez97/01-datos?scriptVersionId=354823890)
corresponden al bloque 1; el pipeline completo se verificó localmente.

## Documentación
* `00_general/metodologia.md` — fundamento, qué se adapta de los artículos y qué se simplifica.
* `00_general/arquitectura.md` — módulos, contrato de archivos, decisiones técnicas.
* `00_general/reproduccion.md` — instrucciones de reproducción desde entorno limpio.
* `00_general/incidencias.md` — problemas encontrados y cómo se resolvieron.
* `04_final/reporte/reporte.tex` / `reporte.pdf` — artículo IEEE de cinco páginas.
* `04_final/presentacion_final.pptx` — presentación editable con notas y responsables.

## Referencias
* M. Aharon, M. Elad, A. Bruckstein, “K-SVD: An Algorithm for Designing Overcomplete Dictionaries
  for Sparse Representation”, IEEE TSP 54(11), 2006. https://doi.org/10.1109/TSP.2006.881199
* M. Elad, M. Aharon, “Image Denoising Via Sparse and Redundant Representations Over Learned
  Dictionaries”, IEEE TIP 15(12), 2006. https://doi.org/10.1109/TIP.2006.881969

Licencia: MIT.
