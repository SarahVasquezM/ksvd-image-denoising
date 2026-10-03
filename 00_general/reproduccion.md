# Reproducción

## Local (pip)
```bash
git clone --branch codex/auditoria-final https://github.com/SarahVasquezM/ksvd-image-denoising.git
cd ksvd-image-denoising
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py                 # bloques 1-4 (≈ 10 s en CPU)
python tools/build_report.py      # reporte (PDF si hay pdflatex) + presentación
python -m pytest                  # 32 pruebas
bash tools/execute_notebooks.sh   # regenera y ejecuta los 4 notebooks en orden
```
Con conda: `conda env create -f environment.yml && conda activate ksvd`.

Bloques sueltos: `python -m src.pipeline --blocks 2 3` o `python 02_modelo/modelo.py`.
Hilos BLAS: `python run_all.py --threads 1` (por defecto 4; `0` = sin límite).

> Si `PYTHONPATH` contiene rutas de otros proyectos (p. ej. ROS) y pytest falla al cargar plugins
> ajenos, ejecutar `env -u PYTHONPATH python -m pytest` (ver `incidencias.md`, I-6).

## Kaggle (CPU)
1. Crear un notebook nuevo con acelerador **None (CPU)** e **Internet activado** (necesario para
   clonar el repositorio; `skimage.data.camera/coins` vienen incluidas en scikit-image).
2. En la primera celda:
   ```python
   !git clone --depth 1 --branch codex/auditoria-final https://github.com/SarahVasquezM/ksvd-image-denoising.git
   %cd ksvd-image-denoising
   !pip install -q threadpoolctl   # normalmente ya está instalado
   !python run_all.py && python -m pytest -q
   ```
3. Alternativamente, subir cualquiera de los notebooks del repositorio: su primera celda detecta
   `/kaggle/working`, clona el repositorio y continúa.

**Estado verificable:** el bloque 1 se ejecutó en Kaggle CPU con estado Successful (versión 1,
runtime reportado de 28 s) y produjo los mismos controles que la ejecución local. El pipeline completo
de los cuatro bloques se validó localmente; todavía no se afirma una ejecución completa en Kaggle.
Recursos privados de Sarah: [notebook 01_datos](https://www.kaggle.com/code/sarahvasquez97/01-datos?scriptVersionId=354823890)
y [dataset v1](https://www.kaggle.com/datasets/sarahvasquez97/ksvd-dani-datos-v1).

## Determinismo
* Semillas fijas (42 muestreo, 43 K-SVD, 44 ruido). `datos.npz` se reproduce **bit a bit**.
* `D`, `A_train` y las reconstrucciones se reproducen a nivel de redondeo (≈ 1e-15) — el orden de
  las operaciones BLAS puede variar con el número de hilos.
* MSE/PSNR se reproducen con diferencias ≤ 4e-15. Los **tiempos** varían entre corridas y máquinas.

## Auditoría final desde entorno limpio (2026-10-03)

### A. Clon limpio + venv nuevo con `pip install -r requirements.txt`
Versiones resueltas: Python 3.11.16, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1,
scikit-image 0.26.0, matplotlib 3.11.2, pandas 3.0.6.

| Paso | Resultado |
|---|---|
| `python -m pytest -q` sobre el clon (artefactos versionados) | **32 passed** en la rama auditada |
| `python run_all.py` | exit 0; integración: todas las comprobaciones `true`, contingencia 0 |
| Comparación artefactos regenerados vs versionados | `datos.npz` idéntico; `D` máx. 9.7e-16; `A_train` 2.2e-15; reconstrucciones 4.4e-16; `resultados.csv` (sin tiempos) 3.6e-15; `historial.csv` (sin tiempos) 0 |
| `bash tools/execute_notebooks.sh` | 4 notebooks ejecutados, 0 errores; reporte y presentación regenerados |

### B. Clon limpio + versiones mínimas (similares a Kaggle)
Python 3.10, numpy 1.24.4, scipy 1.10.1, scikit-learn 1.2.2, scikit-image 0.21.0,
matplotlib 3.7.5, pandas 1.5.3, threadpoolctl 3.1.0.

| Paso | Resultado |
|---|---|
| `python -m pytest -q` (incluye reproducción completa del pipeline en un directorio temporal y comparación con los artefactos versionados) | **31 passed** en la auditoría original; la rama actual añade una prueba de validación de entradas |

Advertencias observadas: avisos de parada temprana de OMP (esperados, ver I-3) y avisos de
deprecación de `pyparsing` dentro de matplotlib 3.7 (ajenos al proyecto).

### Qué cubren las pruebas
* `test_preprocesamiento.py` — formas, orden de recorrido, último inicio sin duplicados, centrado,
  ida y vuelta < 1e-10 en ambas imágenes, no-clipping, cobertura, muestreo reproducible y filtrado,
  configuración congelada.
* `test_modelo.py` — `sparse_code` (formas, N = 1, señales nulas, igualdad con `orthogonal_mp`,
  rechazo de D no normalizado, aviso de parada temprana), **SVD de soporte fijo = mejor rango 1**,
  fórmula literal, soporte y monotonía de la actualización, reinicialización y conservación de átomos,
  contrato y determinismo de `fit_ksvd`.
* `test_evaluacion.py` — métricas de control (MSE 0/PSNR ∞; 0.01/20 dB), ruido reproducible y sin
  clipping, reconstrucción exacta con base completa, respeto de T0.
* `test_integracion.py` — contratos de todos los artefactos versionados y ejecución completa del
  pipeline en un directorio temporal comparada con lo versionado.
