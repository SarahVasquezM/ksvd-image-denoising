# Arquitectura

## Principio: una sola fuente de verdad
Todo el código vive en `src/`. Las carpetas de bloque **no** contienen implementaciones paralelas:
sus `.py` reexportan desde `src/` y sus notebooks llaman a `src/`.

```
src/
├── preprocesamiento.py   # bloque 1: imágenes, extract/assemble, muestreo, datos.npz + config.json
├── modelo.py             # bloque 2: sparse_code (OMP sklearn), K-SVD propio, validación, contingencia
├── evaluacion.py         # bloque 3: métricas, ruido, reconstrucción, run_experiments
├── adaptativo.py         # entrenamiento en la imagen ruidosa y comparación con la ruta externa
├── visualizacion.py      # figuras de todos los bloques
└── pipeline.py           # orquestación run_block1..4 y CLI (python -m src.pipeline)
tools/
├── build_notebooks.py    # genera los notebooks (fuente) — luego se ejecutan con nbconvert
└── build_report.py       # reporte LaTeX/PDF + Markdown y presentación PDF desde los resultados
run_all.py                # atajo: ejecuta los bloques 1-4
run_comparative.py        # genera la variante adaptativa en 05_comparacion/
```

## Flujo de datos (contrato por archivos)
```
skimage.data ──► [1] build_dataset ──► 01_datos/datos.npz, config.json
                                            │
                                            ▼
                                  [2] train_from_files ──► 02_modelo/modelo.npz, historial.csv, config_efectiva.json
                                            │
                                            ▼
                                  [3] run_experiments ──► 03_evaluacion/resultados.csv, reconstrucciones.npz
                                            │
                                            ▼
                                  [4] run_block4 + build_report ──► 04_final/figuras, reporte/, presentacion/

01_datos/datos.npz ──► ruido reproducible ──► parches de coins ruidosa
                                                    │
                                                    ▼
                                      train_adaptive_from_files
                                                    │
                                                    ▼
                 05_comparacion/adaptativo/modelo_adaptativo.npz
                                                    │
                                                    ▼
                 evaluación común + comparación con 03_evaluacion
                                                    │
                                                    ▼
                 05_comparacion/comparacion_resultados.csv y figuras/
```
Cada bloque puede regenerarse aislado mientras existan los archivos del bloque anterior
(`python -m src.pipeline --blocks 3`).

La extensión comparativa no modifica los paquetes aceptados. `python run_comparative.py` lee los
productos de los bloques 1--3, aprende el segundo diccionario y escribe exclusivamente en
`05_comparacion/`. Con `--rebuild-base` se regeneran primero los bloques 1--3.

## API pública
| Función | Firma | Notas |
|---|---|---|
| `extract_patches` | `(image, patch_size=8, stride=4) -> Z (64,N), means (N,), positions (N,2)` | parches centrados |
| `assemble_patches` | `(patches, positions, image_shape, patch_size=8)` | parches con media restaurada; sin clipping |
| `sparse_code` | `(D, Z, max_nonzero) -> A (K,N)` | `orthogonal_mp`; señales ~0 → 0 |
| `fit_ksvd` | `(X_train, n_atoms=128, n_iter=8, train_sparsity=4, seed=43) -> D, A_train, history` | |
| `ksvd_dictionary_update` | `(X, D, A) -> D, A, info` | una pasada átomo por átomo |
| `run_experiments` | `(data_path, model_path, config_path, output_dir) -> DataFrame` | |
| `train_adaptive_from_files` | `(data_path, config_path, output_dir) -> dict` | usa parches de la observación ruidosa |
| `run_comparative` | `(root, n_timing_repeats=5) -> dict` | compara ambas rutas sin sobrescribir la externa |

## Decisiones técnicas
* **Hilos BLAS limitados a 4** (`threadpoolctl`) en pipeline y notebooks: con matrices 64×1500, más
  hilos empeoran el tiempo por sobresuscripción (medido: 9–12 s con 24 hilos vs ≈ 1.5 s con 1–4).
  No altera los resultados más allá de redondeo (diferencia máxima en `D` ≈ 1e-15).
* **Advertencias de OMP:** se registran y se re-emite un resumen; no se silencian.
* **Notebooks generados por script** (`tools/build_notebooks.py`) para que su contenido sea revisable
  en un diff y reproducible; se versionan **ejecutados** (con salidas).
