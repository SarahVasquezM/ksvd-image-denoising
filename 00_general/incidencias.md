# Registro de incidencias y decisiones

Cada entrada: problema → diagnóstico → corrección → efecto en resultados.

## I-1. Instrucciones del proyecto incompletas a partir de la sección 9
* **Problema:** el texto con los requisitos se cortó a mitad de la sección 9 (evaluación), en la línea
  `noise = rng.normal(0, 20/255, size=test`.
* **Decisión:** se completó lo faltante con los criterios más directos y se documentó cada supuesto:
  * `size=test_image.shape`, ruido aditivo `noisy = test_image + noise`, **sin clipping**;
  * métricas siempre contra la imagen de prueba limpia; se reporta también la ruidosa sin procesar;
  * tiempo de reconstrucción = mediana de 5 repeticiones (extracción + OMP + reensamblado);
  * productos de `03_evaluacion/` y `04_final/` según la estructura de carpetas pedida;
  * reporte en LaTeX/PDF y presentación en PDF, ambos generados desde los archivos de resultados.
* **Efecto:** si las instrucciones originales piden otra cosa (p. ej. recortar la imagen ruidosa a
  [0, 1]), basta cambiar `make_noisy` en `src/evaluacion.py` y volver a ejecutar
  `python -m src.pipeline --blocks 3 4`.

## I-2. Sobresuscripción de hilos BLAS
* **Problema:** el entrenamiento completo tardaba 9–12 s con tiempos por iteración erráticos
  (0.3–2.8 s).
* **Diagnóstico:** OpenBLAS usaba 24 hilos para matrices de 64 × ~1500; con 1 o 4 hilos el mismo
  entrenamiento tarda ≈ 1.5 s.
* **Corrección:** `src/pipeline.py`, los wrappers y los notebooks limitan los hilos a 4 con
  `threadpoolctl` (`--threads 0` quita el límite). Los hilos usados quedan en
  `config_efectiva.json → threadpools`.
* **Efecto:** sólo en tiempos. Diferencia máxima en `D` entre 1 hilo y 24 hilos ≈ 8.6e-16.

## I-3. Advertencias de OMP en la primera iteración
* **Observación:** `orthogonal_mp` avisa que terminó antes de `T = 4` átomos en 128 señales en la
  iteración 1 y al codificar con `D_init`.
* **Diagnóstico:** esas 128 señales son los parches elegidos como átomos iniciales: se representan
  exactamente con un átomo (residual cero). Es el comportamiento correcto de `max_nonzero` como máximo.
* **Corrección:** ninguna en el algoritmo. `sparse_code` resume las advertencias de scikit-learn en
  una sola por llamada (con el conteo) en lugar de suprimirlas; otras advertencias pasan intactas.

## I-4. Contingencia de entrenamiento
* Se midió primero la configuración deseada (K = 128, 1500 parches, 8 iteraciones) con un
  presupuesto de 3600 s. Tardó ≈ 1.5 s. **No se aplicó ninguna contingencia** (nivel 0).
  `config_efectiva.json` registra el intento (`attempts`) y `contingency_applied = false`.
* El código de contingencia (niveles 1 y 2) existe y se activa sólo si el tiempo proyectado tras la
  primera iteración excede el presupuesto; si se llegara a K = 64, `overcomplete = false` y la
  etiqueta lo dice explícitamente.

## I-5. Paquetes de LaTeX ausentes
* **Problema:** `pdflatex` falló con `hyperref` (falta `etoolbox.sty`); tampoco hay `babel-spanish`,
  `booktabs` ni `beamer` en la instalación local.
* **Corrección:** el reporte usa sólo paquetes básicos (`url` en lugar de `hyperref`; nombres de
  figura/tabla redefinidos a mano) y la presentación se genera con matplotlib (PDF 16:9).
  Si `pdflatex` no existe, `tools/build_report.py` conserva `reporte.tex` y `reporte.md`.

## I-6. Plugin de pytest ajeno al proyecto
* **Problema:** en la máquina de desarrollo `PYTHONPATH` apunta a ROS (`/opt/ros/...`) y pytest
  carga el plugin `launch_testing`, que es incompatible y aborta antes de correr pruebas.
* **Corrección:** ejecutar con `env -u PYTHONPATH` (o en un entorno sin esa variable). No afecta al
  código del proyecto; se documenta en `reproduccion.md`.

## I-7. Verificación de afirmaciones sobre el artículo
* La descarga vía DOI de IEEE no devolvió contenido; el PDF del artículo de denoising se obtuvo de la
  página del autor y se convirtió a texto para verificar los datos citados (8×8, 64×256, `C = 1.15`,
  `J = 10`, `λ = 30/σ`, DCT redundante, SVD sobre los ejemplos que usan el átomo). Del artículo de
  K-SVD sólo se usan afirmaciones generales sobre el algoritmo.
