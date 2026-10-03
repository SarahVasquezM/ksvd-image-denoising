# K-SVD: datos y parches

Proyecto académico de cuatro integrantes. Este repositorio contiene la entrega
comprobada de **Dani (integrante 1)**: preparación de imágenes, extracción y
ensamblado de parches. No contiene todavía modelo, entrenamiento ni resultados
de denoising. Objetivo del equipo: estudiar reconstrucción dispersa y eliminación
de ruido con K-SVD; adaptación didáctica, no réplica completa del artículo.

## Ejecutar localmente

Python 3.12 fue utilizado. Desde la raíz del repositorio:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/ejecutar_notebook.py
```

En Linux/Kaggle el ejecutable equivalente es `python` (o `.venv/bin/python`
para un entorno local). En Kaggle comprobar primero los imports: no instalar
estas versiones sobre un entorno que ya funciona. Cada ejecución registra sus
versiones reales. `scripts/ejecutar_notebook.py` inicia un kernel limpio y exporta
el notebook ejecutado, artefactos y manifiesto a `artifacts/entrega_01/`, además
de `artifacts/entrega_01_v1.zip`. Es una salida regenerable; congelar una copia
antes de regenerar una versión aceptada.

## Estructura y responsabilidades

| Integrante | Bloque | Fuente/entrega |
|---|---|---|
| Dani (1) | Entorno, imágenes, parches | `src/preprocesamiento.py`, `notebooks/01_datos.ipynb` |
| 2 | OMP, K-SVD, entrenamiento | Futuro `modelo.py`, `02_modelo.ipynb`, modelo/historial |
| 3 | Ruido, reconstrucciones, métricas | Futuro `evaluacion.py`, `03_experimentos.ipynb` |
| 4 | Integración, reporte, presentación | Futuro notebook final y documentos |

`docs/` conserva los seis documentos recibidos, el contrato y la guía de trabajo.
`tests/` protege orientación, bordes, promedio, filtro, índices y datos reales.
`src/preprocesamiento.py` es la fuente de verdad; no copiar sus funciones a celdas.

## Contrato y resultados comprobados

camera: 256×256; coins: 128×128. Conversión float64 antes de resize. Parches 8×8,
paso 4, orden C, columnas centradas sin normalización de norma. Selección de 1500
columnas con norma >1e-8 y default_rng(42), sin reemplazo. Índices sobre la
extracción completa. Todos los parámetros compartidos se exportan a config.json.

Siete pruebas unitarias aprobadas y notebook completo ejecutado en kernel nuevo
localmente. Extracción: 3969 parches de camera y 961 de coins. Error máximo de
ida/vuelta en ambas: 1.1102230246251565e-16; cobertura de 1 a 4. Mayor media
absoluta seleccionada: 6.904199434387692e-16. Ver `comprobaciones.json` de cada
paquete para evidencia y entorno de origen. Esto verifica preparación, no denoising.

Ver [colaboración](docs/colaboracion.md), [contrato](docs/contrato.md) y
[estado de plataformas](docs/estado_plataformas.md).

## Referencias

- Elad y Aharon (2006), [Image Denoising Via Sparse and Redundant Representations Over Learned Dictionaries](https://doi.org/10.1109/TIP.2006.881969).
- Aharon, Elad y Bruckstein (2006), [K-SVD](https://doi.org/10.1109/TSP.2006.881199).
- [Imágenes de scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html).
- [resize](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize).
- [Kaggle Notebooks](https://www.kaggle.com/docs/notebooks).

No se seleccionó licencia de redistribución para el proyecto ni se relicenciaron
las imágenes. El artículo adjunto se consultó localmente y no se redistribuye.
