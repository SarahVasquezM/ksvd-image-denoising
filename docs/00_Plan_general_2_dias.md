# Plan mínimo de K-SVD: cuatro integrantes, dos días

## Decisión y alcance

La propuesta de reducir datos, usar librerías y hacer una demostración visual es adecuada. El cambio necesario es conservar K-SVD en el código que produce los resultados: ejecutar DictionaryLearning y explicar K-SVD mediante pseudocódigo no satisface por sí solo «implementar K-SVD». Una celda conceptual sirve para enseñar, pero no reemplaza el algoritmo ejecutado.

Implementaremos únicamente el ciclo K-SVD: codificación con OMP ya disponible, actualización restringida de cada átomo con SVD ya disponible y repetición. No implementaremos OMP ni la SVD numérica desde cero. No añadiremos una segunda implementación de dictionary learning.

Objetivo: aprender un diccionario y comparar una imagen limpia y una observación ruidosa con sus respectivas reconstrucciones dispersas. No hay promesa de superar otros métodos ni de reproducir los valores del artículo.

## Configuración congelada

| Elemento | Decisión |
|---|---|
| Plataforma | Kaggle, CPU; un notebook de trabajo por persona y un notebook final del integrante 4 |
| Datos | `skimage.data.camera()` para entrenamiento; `skimage.data.coins()` para prueba |
| Preparación | `img_as_float64`, luego `resize` con `anti_aliasing=True`, `preserve_range=True`; registrar que hay redimensionamiento |
| Imágenes finales | Entrenamiento 256×256; prueba 128×128; grises y rango [0,1] |
| Parches | 8×8; paso 4; orden por filas, luego columnas; aplanado C |
| Centrado | Restar media por parche y conservarla; no dividir cada parche entre su norma |
| Muestras de entrenamiento | 1,500 parches centrados no constantes, sin reemplazo; RNG 42 |
| Diccionario | 128 átomos de longitud 64; columnas unitarias |
| Entrenamiento | 8 iteraciones, máximo 4 átomos por parche; RNG 43 |
| Prueba | Máximo 2, 4 y 8 átomos por parche; mismo diccionario congelado |
| Ruido | Una realización gaussiana, sigma=20/255; RNG 44 |
| Métricas | MSE, PSNR, media de coeficientes activos y tiempo de reconstrucción |
| Referencia mínima | Imagen ruidosa sin procesar; sin DCT ni otros métodos en esta entrega |
| Resultados | Tres reconstrucciones limpias y tres ruidosas, siete filas de métricas contando referencia |

Se usan dos imágenes para evitar entrenar con la versión limpia exacta de la imagen que se evalúa. Eso no convierte el estudio en una evaluación generalizable: sigue siendo una prueba de concepto con una sola imagen de evaluación, una semilla de entrenamiento y una realización de ruido. No hay conjunto de validación; no se ajustan parámetros mirando la prueba. Se reportan los tres valores fijados de antemano.

La figura principal usa T0=4, fijado antes de ejecutar. El barrido completo se muestra aunque otro valor resulte mejor. No dibujar una curva de PSNR necesariamente creciente: en eliminación de ruido puede caer al aumentar los coeficientes.

## Responsables y entregas únicas

| Persona | Único bloque de responsabilidad | Recibe | Entrega a | Límite |
|---|---|---|---|---|
| 1: Sarah | Kaggle, datos, extracción/ensamblado y contrato comprobado | Este plan | 2, 3 y 4 | Día 1, 12:00 |
| 2 | OMP, K-SVD y entrenamiento | Paquete de Sarah | 3 y 4 | Día 1, 17:00 |
| 3 | Ruido, reconstrucciones, métricas e interpretación | Paquetes 1 y 2 | 4 | Día 2, 12:00 |
| 4 | Integración final, figuras, reporte y presentación | Entregas anteriores | Equipo | Día 2, 17:00 |

Estimación de trabajo individual enfocado, además de lectura y ensayo: Sarah 4–5 h; persona 2 5–6 h; persona 3 4–5 h; persona 4 5–6 h. Son presupuestos, no mediciones ni garantías. El entrenamiento puede requerir espera adicional. La persona 4 recibe texto técnico listo de las otras tres y no redacta todo desde cero.

## Cronograma

### Día 1

| Hora | Sarah | Persona 2 | Persona 3 | Persona 4 |
|---|---|---|---|---|
| 08:30–09:15 | Todos: leer objetivo y fundamentos, acordar contratos y comprobar cuentas |
| 09:15–11:45 | Kaggle, imágenes, parches y pruebas | Desarrollar K-SVD sobre una matriz sintética 64×200 | Preparar ruido, métricas y ensamblado usando la API acordada y ejemplos sintéticos | Estructura del notebook, reporte y diapositivas; fundamentos |
| 11:45–12:00 | Validar y entregar paquete 1 | Comprobar carga del paquete | Comprobar formas del paquete | Archivar versión recibida |
| 12:00–16:30 | Bloque técnico terminado | Entrenamiento real y documentación | Preparar runner que recibirá modelo; comprobaciones de métricas | Incorporar datos y preparar figuras con datos de ejemplo claramente marcados |
| 16:30–17:00 | — | Validar y entregar paquete 2 | Cargar modelo y ejecutar primer caso | Incorporar módulo y explicación del método |
| 17:00–18:00 | — | Bloque técnico terminado | Verificar primera reconstrucción; guardar avance | Actualizar borrador sin inventar resultados |

### Día 2

| Hora | Actividad |
|---|---|
| 09:00–11:30 | Persona 3 ejecuta los seis casos, verifica resultados y redacta análisis. Persona 4 termina estructura y antecedentes. |
| 11:30–12:00 | Persona 3 entrega paquete 3 completo. |
| 12:00–15:30 | Persona 4 integra y genera las tres figuras; completa reporte y presentación con los textos recibidos. |
| 15:30–16:30 | Persona 4 reinicia el entorno y ejecuta el notebook final; puede cargar el modelo guardado por defecto. |
| 16:30–17:00 | Exportar y entregar notebook, modelo, resultados, reporte y diapositivas. |
| 17:00–18:00 | Todos: revisar afirmaciones, ensayar 13 min y comprobar respaldo descargado. |
| 18:00–19:00 | Margen para un bloqueo real; no nuevas funciones. |

La cadena de datos real es Sarah → 2 → 3 → 4. La escritura de código, los fundamentos y la estructura documental sí pueden prepararse en paralelo gracias a los contratos. No es necesario que una persona termine todo para que otra empiece a pensar o programar.

## Contrato técnico obligatorio

Todas las matrices numéricas se guardan en float64, sin NaN ni infinito; las posiciones son enteras. `Z` contiene señales por columnas. No cambiar a señales por filas sin un adaptador explícito.

### Módulo de Sarah: `preprocesamiento.py`

- `extract_patches(image, patch_size=8, stride=4) -> (Z, means, positions)`.
- `Z`: (64,N), parches centrados; `means`: (N,); `positions`: (N,2), fila y columna superior izquierda.
- Incluir último inicio `H-8`/`W-8` si el paso no llega exactamente; sin duplicar posiciones. En las imágenes fijadas no hace falta padding.
- `assemble_patches(patches, positions, image_shape, patch_size=8) -> image`.
- `patches`: (64,N), YA con media restituida. Sumar contribuciones y dividir por conteo; todos los píxeles deben tener cobertura positiva. No recortar intensidades en esta función.
- Reconstrucción de control: `assemble_patches(Z + means[None,:], positions, image.shape)` recupera la entrada con error máximo <1e-10.

### Paquete 1

- `datos.npz`: `train_image` (256,256), `test_image` (128,128), `X_train` (64,1500), `train_indices` (1500,).
- `train_indices` indexa las columnas originales de la extracción completa de train_image, antes del filtro de parches constantes.
- `config.json`: version_contrato=1; patch_size=8; stride=4; n_atoms=128; n_iter=8; train_sparsity=4; eval_sparsities=[2,4,8]; sigma=20/255; seeds={data:42,model:43,noise:44}; image names y tamaños.
- `preprocesamiento.py`, `01_datos.ipynb`, `versiones.txt`, `nota_datos.md`.

### Módulo persona 2: `modelo.py`

- `sparse_code(D, Z, max_nonzero) -> A`, formas (64,K), (64,N), (K,N).
- `fit_ksvd(X_train, n_atoms=128, n_iter=8, train_sparsity=4, seed=43) -> (D, A_train, history)`.
- A_train se recalcula con el D FINAL antes de devolver; history registra qué error corresponde a cada fase.
- `modelo.npz`: `D` (64,K), `D_init` (64,K), `A_train` (K,N_train).
- `historial.csv`: iteration, train_mse, elapsed_s, unused_atoms.
- Entregar `02_modelo.ipynb`, `modelo.py`, `modelo.npz`, `historial.csv`, `nota_metodo.md`, `config_efectiva.json`.
- Todos los consumidores toman K y N_train de los arrays y los parámetros finales de config_efectiva.json.

### Paquete persona 3

- `evaluacion.py`: `run_experiments(data_path, model_path, config_path, output_dir)`; usa módulos 1 y 2.
- `resultados.csv`: case_id, input_kind, method, sigma, max_nonzero, mse, psnr_db, mean_nonzero, reconstruction_s, noise_seed.
- IDs: noisy_baseline, clean_t2, clean_t4, clean_t8, noisy_t2, noisy_t4, noisy_t8.
- Valores: input_kind=clean/noisy; method=identity/ksvd; baseline max_nonzero=0, mean_nonzero y reconstruction_s vacíos.
- `reconstrucciones.npz`: `original`, `noisy` y las seis claves clean_t2..noisy_t8, todas (128,128).
- `03_experimentos.ipynb`, `evaluacion.py`, archivos anteriores y `nota_resultados.md`.

### Política de evaluación

Crear ruido después del redimensionamiento. No recortar `noisy` antes de procesarla. OMP sólo recibe sus parches; no recibe la referencia limpia para corregirla. Para el experimento limpio sí se usan los parches limpios como entrada.

Todas las métricas se calculan sobre arrays flotantes SIN clipping y con rango de referencia 1.0. Guardar también mínimos/máximos en la nota. Limitar visualización a [0,1] mediante vmin/vmax; no modificar arrays para mejorar métricas. La política es la misma para los seis resultados.

Tiempo de reconstrucción: extracción + OMP + ensamblado, sin entrenamiento, archivos ni gráficos. Contar coeficientes activos con abs(A)>1e-10. El entrenamiento se reporta por separado.

## Entrega entre notebooks en Kaggle

No editar el mismo notebook a la vez. Cada autor usa su notebook; cada paquete queda en `/kaggle/working/entrega_0N/`. Guardar una versión con salidas, descargar el paquete y compartirlo por el canal privado acordado. El receptor lo adjunta como datos privados a su notebook o usa salidas versionadas del notebook anterior si esa opción está disponible.

Las entradas se leen desde la ruta real montada en `/kaggle/input/`; las salidas se escriben en `/kaggle/working/`. No asumir el nombre del directorio: localizarlo y asignarlo a una variable de configuración. No sobrescribir una versión aceptada; congelar paquetes. Guardar copia descargada antes de terminar la sesión.

Persona 4 mantiene la lista de versiones aceptadas y el notebook final. Los notebooks incluyen el código por secciones; los `.py` son la fuente compartida y no se mantienen copias divergentes.

## Reducción de emergencia

Si persona 2 observa a las 14:30 del día 1 que no podrá terminar: reducir a K=64, 500 primeras columnas de X_train y 5 iteraciones; publicar config_efectiva.json junto con el modelo. La selección de 500 columnas ya es reproducible. Cambia el número de átomos, no la API. K=64 deja de ser sobrecompleto, y así se debe declarar. Primero conservar K=128 y bajar sólo parches/iteraciones si alcanza.

Si persona 3 tiene problemas de tiempo: terminar primero clean_t4, noisy_t4 y baseline. Marcar barrido incompleto; persona 4 cambia figura 3 a comparación de esos métodos, sin inventar puntos. No añadir DCT, SSIM, más imágenes o tipos de ruido en el margen final.

Si K-SVD no funciona: reparar el núcleo o comunicar al profesor que se propone un cambio de alcance. DictionaryLearning es un plan alternativo sujeto a aceptación del profesor, no un cumplimiento equivalente.

## Entrega académica

Tres figuras: original/ruidosa/reconstruida limpia/reconstruida ruidosa (T0=4); mosaico del diccionario final; PSNR frente a máximo de átomos por parche con curvas limpia y ruidosa y referencia ruidosa horizontal. Acompañar tabla de siete filas, o tabla de tres si se activó contingencia.

Reporte máximo 5 páginas con referencias: objetivo/fundamento 0.75; datos 0.75; proceso 1.5; resultados/análisis 1.5; cierre/referencias 0.5. Cada autor entrega su texto: Sarah datos y objetivo provisional; 2 método; 3 resultados y limitaciones; 4 integra fundamento, conclusiones y formato.

Presentación de 8 diapositivas, 13 min: 1–2 objetivo/datos (Sarah, 3 min); 3–4 representación/K-SVD (2, 4 min); 5–6 experimentos/resultados (3, 3 min); 7–8 discusión/conclusiones (4, 3 min). Reservar 2 min bajo el máximo de 15.

Aceptar la entrega cuando: hay código K-SVD ejecutado, modelo guardado, original y reconstrucciones, métricas reales, archivos reproducibles, reporte ≤5 páginas y exposición ensayada. Una mejora limitada o un resultado negativo también puede analizarse honestamente.

## Referencias y recursos

Fundamento científico:
- Aharon, Elad y Bruckstein (2006). K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation. https://doi.org/10.1109/TSP.2006.881199
- Elad y Aharon (2006). Image Denoising Via Sparse and Redundant Representations Over Learned Dictionaries. https://doi.org/10.1109/TIP.2006.881969
- PDF de los autores: https://elad.cs.technion.ac.il/wp-content/uploads/2018/02/KSVD_Denoising_IEEE_TIP.pdf

Documentación de herramientas (no reemplaza los papers como fundamento):
- Kaggle: https://www.kaggle.com/docs/notebooks
- OMP: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html
- SVD: https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
- Imágenes: https://scikit-image.org/docs/stable/api/skimage.data.html
- Métricas: https://scikit-image.org/docs/stable/api/skimage.metrics.html
