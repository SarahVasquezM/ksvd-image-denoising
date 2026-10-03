# Integrante 3 — Reconstrucción, ruido, métricas y análisis

## Tu responsabilidad y plazo

Aplicar el diccionario ya entrenado, ejecutar el experimento congelado y entregar arrays, tabla y análisis. Empiezas día 1 a las 09:15 con ejemplos sintéticos; recibes datos a las 12:00, modelo a las 17:00 y entregas resultados el día 2 a las 12:00. Carga estimada: 4–5 horas efectivas más lectura y ensayo.

No entrenas nuevos modelos ni cambias datos/hiperparámetros mirando los resultados. No necesitas esperar al modelo para escribir ruido, métricas, guardado y el runner.

## Herramientas y funciones

Kaggle CPU; NumPy, scikit-image.metrics, pandas/csv, time. Importar `extract_patches`, `assemble_patches` del módulo de Dani y `sparse_code` del módulo de persona 2.

Entregar `evaluacion.py` con `run_experiments(data_path, model_path, config_path, output_dir)`. Lee parámetros de config_efectiva.json y tamaños de los arrays, no de supuestos repetidos.

## Preparación en paralelo: día 1, 09:15–12:00

1. Escribir cálculo de MSE y PSNR con data_range=1.0.
2. Comprobar métricas con imagen contra sí misma: MSE=0 y PSNR infinito, caso válido. Una diferencia constante de 0.1 debe dar MSE=0.01 y PSNR=20 dB.
3. Escribir generación de ruido mediante default_rng(44).normal(0,20/255,size=image.shape).
4. Preparar esquema CSV y NPZ con las claves acordadas.
5. Escribir el runner contra las firmas acordadas; si usas funciones ficticias para depurarlo, márcalas como tales y elimínalas antes de la ejecución real.

## Recepción y ejecución

A las 12:00 del día 1 verifica las funciones de Dani con la imagen real. A las 17:00 verifica carga de D y ejecuta primero clean_t4 y noisy_t4. Guarda un avance. Día 2 completa el resto antes de las 11:30 y entrega a las 12:00.

Para cada entrada (original limpia o noisy):

1. Extraer Z, means, positions de ESA entrada.
2. A = sparse_code(D,Z,max_nonzero=T0).
3. patches_hat = D@A + means[None,:].
4. image_hat = assemble_patches(patches_hat,positions,image.shape).
5. Comparar image_hat con la original limpia, que sólo interviene como referencia de evaluación en el caso ruidoso.

Crear noisy una sola vez y reutilizarla para T0=2,4,8. No generar un ruido distinto por configuración. No recortar noisy ni los arrays de reconstrucción antes de las métricas. No usar la media del parche limpio para restaurar una entrada ruidosa.

Medir tiempo desde extracción hasta ensamblado, excluyendo entrenamiento y gráficos. Contar coeficientes con abs(A)>1e-10. Mantener D sin cambios durante toda la evaluación.

## Experimento y tabla final

Siete casos: noisy_baseline; clean_t2, clean_t4, clean_t8; noisy_t2, noisy_t4, noisy_t8.

CSV con columnas exactas: case_id,input_kind,method,sigma,max_nonzero,mse,psnr_db,mean_nonzero,reconstruction_s,noise_seed.

- noisy_baseline: input_kind=noisy, method=identity, sigma=20/255, max_nonzero=0, noise_seed=44; actividad y tiempo vacíos.
- clean_t*: input_kind=clean, method=ksvd, sigma=0; noise_seed vacío.
- noisy_t*: input_kind=noisy, method=ksvd, sigma=20/255, noise_seed=44.

`reconstrucciones.npz`: original, noisy y seis reconstrucciones por su case_id. Todas tienen forma (128,128), float64.

## Qué analizar

Escribir `nota_resultados.md` de 350–450 palabras con números reales:

- Cómo cambia PSNR al aumentar T0 en entradas limpias y ruidosas.
- Delta PSNR de cada resultado ruidoso respecto de noisy_baseline.
- Si hay suavizado excesivo, textura perdida, ruido residual o artefactos de parches.
- Cuál es el costo de reconstrucción, separado del entrenamiento.
- Limitaciones: una imagen de prueba, una realización de ruido y parámetros prefijados; no evidencia general ni comparación con el estado del arte.
- Qué diferencia hay entre reconstrucción limpia y denoising.

El panel principal se fija en T0=4; se puede comentar otro valor del barrido sin presentarlo como selección validada. Si el resultado no mejora, reportarlo; revisar fallos de implementación, no cambiar silenciosamente imagen o semilla para conseguir una figura bonita.

## Comprobaciones y entrega

- [ ] No se ha entrenado con coins ni con sus parches limpios.
- [ ] Los seis casos comparten el mismo D y las tres entradas ruidosas el mismo ruido.
- [ ] Las métricas se calculan con floats sin clipping y data_range=1.
- [ ] No hay NaN ni infinito en imágenes; PSNR infinito sólo si MSE=0.
- [ ] El número de coeficientes activos respeta cada límite.
- [ ] Resultados guardados sin redondear prematuramente.
- [ ] Se incluye el rango mínimo/máximo observado y configuración efectiva en la nota.

Entregar `03_experimentos.ipynb`, `evaluacion.py`, `resultados.csv`, `reconstrucciones.npz`, `nota_resultados.md` y registro de incidencias si existe. La persona 4 hace las figuras de presentación; tú no tienes que volver a producirlas.

Si el día 2 a las 10:30 hay un bloqueo, prioriza baseline, clean_t4 y noisy_t4 y entrega explícitamente un experimento reducido. No rellenar los casos faltantes con estimaciones.

Tu bloque termina a las 12:00 del día 2. Prepara 3 minutos sobre diseño experimental y resultados para el ensayo.

## Reglas comunes

- Proyecto: reconstrucción y eliminación de ruido con K-SVD. Adaptación didáctica; no réplica completa del paper.
- CPU, Python y Kaggle. No GPU, RGB, grandes datasets ni búsqueda extensa de hiperparámetros.
- K-SVD real: OMP de scikit-learn y SVD de NumPy, unidos por un ciclo propio de actualización de átomos. DictionaryLearning no sustituye ese ciclo.
- Leer `00_Plan_general_2_dias.md`: contiene los contratos obligatorios. Los formatos no se cambian unilateralmente.
- Fechas propuestas: día 1, sábado 3 de octubre de 2026; día 2, domingo 4 de octubre. Horario de Puebla/CDMX. Si empiezan después, desplacen todos los hitos conservando los intervalos. Son plazos internos, no una fecha de entrega del profesor conocida.
- Cada integrante desarrolla un bloque continuo y hace una entrega final. Documenta su bloque en esa misma entrega. Después sólo participa en revisión conceptual y ensayo; no se le programa otra ronda de desarrollo.
- No se promete ausencia absoluta de correcciones. Un bloqueo que impida ejecutar se comunica de inmediato; la persona que recibe integra ajustes menores y el autor sólo atiende un defecto crítico de su bloque.
- Todos leen fundamentos al inicio y deben explicar señal, diccionario, coeficientes, OMP, SVD y evaluación.
- Guardar resultados reales. Los datos de ejemplo utilizados para desarrollar no se incluyen como evidencia experimental.

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
