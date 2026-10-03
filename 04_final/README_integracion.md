# Bloque 4 — Integración final

**Responsable original:** integrante 4. **Notebook:** `04_final.ipynb`.
**Código:** `src/pipeline.py::run_block4`, `src/visualizacion.py`, `tools/build_report.py`.

## Qué hace
1. Lee **sólo desde disco** los productos de los bloques 1–3 (`datos.npz`, `config.json`,
   `modelo.npz`, `historial.csv`, `config_efectiva.json`, `resultados.csv`, `reconstrucciones.npz`).
2. Verifica la coherencia entre bloques y escribe `verificacion_integracion.json`:

   | comprobación | significado |
   |---|---|
   | `K_matches_effective_config` | `D` tiene los átomos declarados en `config_efectiva.json` |
   | `K_matches_results` | `resultados.csv` se generó con ese mismo `D` |
   | `A_train_reproducible` | recodificar `X_train` con el `D` guardado reproduce `A_train` |
   | `history_rows_match_n_iter` | el historial tiene una fila por iteración efectiva |
   | `test_image_matches` | la imagen de referencia de la evaluación es la de `datos.npz` |
   | `eval_sparsities_present` | están todos los T0 de `config.json` |
   | `contingency_level` | nivel de contingencia aplicado (0 = ninguno) |

   Cualquier fallo lanza una excepción: el notebook no termina en silencio con datos incoherentes.
3. Genera las figuras finales (`figuras/`), la tabla final (`tabla_final.csv`), el reporte
   (`reporte/reporte.pdf`, `reporte.tex`, `reporte.md`) y la presentación
   (`presentacion/presentacion.pdf`, `presentacion.md`).

La versión auditada añade entregables editables y revisados visualmente:

* `reporte_final.docx` y `reporte_final.pdf` — cinco páginas, dentro del máximo acordado;
* `presentacion_final.pptx` y `presentacion_final.pdf` — ocho diapositivas, con tabla y gráficos
  nativos editables;
* `../AUDITORIA.md` — dictamen, correcciones y límites de la evidencia.

Todos los números del reporte y la presentación se leen de los archivos de resultados en el momento
de generarlos (`tools/build_report.py`); ninguno está escrito a mano. Las frases de conclusión que
dependen de los datos (p. ej. qué T0 es mejor, si la curva es monótona) también se calculan.

## Cómo regenerar sólo este bloque
```bash
python -m src.pipeline --blocks 4
python tools/build_report.py      # requiere pdflatex para el PDF del reporte
python tools/build_final_report.py  # genera el DOCX final desde los artefactos medidos
```

## Estado de la integración (última ejecución)
Todas las comprobaciones en `true`, `contingency_level = 0`, `D` de forma (64, 128).
