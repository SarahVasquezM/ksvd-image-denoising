# Auditoría técnica de la entrega K-SVD

**Fecha:** 3 de octubre de 2026

**Rama:** `codex/auditoria-final`
**Alcance:** código, contratos entre bloques, datos, artefactos numéricos, notebooks y entregables.

## Dictamen

El proyecto implementa correctamente la adaptación didáctica acordada: OMP se toma de scikit-learn,
la actualización K-SVD usa SVD sobre el residual restringido, las señales se organizan por columnas y
los resultados se obtienen con CPU y `float64`. La ejecución local completa es reproducible dentro de
tolerancias de redondeo. No se encontraron resultados inventados ni uso de la imagen de prueba para
entrenamiento.

Los datos del proyecto ya eran los mismos de Sarah: `camera` para entrenamiento y `coins` para prueba.
La comparación elemento a elemento confirmó igualdad exacta de `train_image`, `test_image`, `X_train`
y `train_indices` con el paquete `entrega_01`; por eso no se añadió un dataset duplicado.

## Evidencia local

- `python run_all.py`: bloques 1 a 4 terminados; integración con todas las comprobaciones verdaderas.
- `pytest -q`: **32 pruebas aprobadas**. Las 12 advertencias de OMP son esperadas cuando una señal ya se
  representa exactamente antes de alcanzar el máximo de cuatro átomos.
- Cuatro notebooks regenerados y ejecutados en kernel nuevo, sin errores.
- Datos: `X_train` de forma `(64, 1500)`, diccionario `D` de forma `(64, 128)` y seis reconstrucciones
  de forma `(128, 128)`.
- Reconstrucción de control extraer/ensamblar: error máximo `1.11e-16` en ambas imágenes.
- Entrenamiento sin contingencia: MSE `9.549e-4` con el diccionario inicial y `3.413e-4` con el final.
- Mejor caso ruidoso: `T0=4`, PSNR `26.12 dB`, mejora de `4.03 dB` frente a la entrada ruidosa.

## Correcciones aplicadas

1. **Validación del bloque de Sarah.** Se conservaron las firmas compartidas y se añadieron controles
   de enteros, finitud, dimensiones, posiciones, cobertura y parámetros inválidos. Se mantuvo el
   ensamblado sin clipping y se añadió una prueba específica de entradas inválidas.
2. **Contrato de evaluación.** El repositorio recibido usaba nombres distintos a los acordados en
   `docs/00_Plan_general_2_dias.md`. Se añadieron las columnas contractuales (`case_id`, `input_kind`,
   `method`, `sigma`, `max_nonzero`, `mean_nonzero`, `reconstruction_s`, `noise_seed`) y las claves
   `original`, `noisy`, `clean_t2..8`, `noisy_t2..8`, conservando los aliases anteriores para no romper
   el reporte ni los notebooks.
3. **Configuración compartida.** `config.json` conserva los nombres usados por el pipeline y añade los
   aliases del contrato original (`n_train`, `constant_threshold`, semillas `data/model/noise`).
4. **Procedencia y reproducción.** Los notebooks y las instrucciones apuntan al repositorio de Sarah.
   Se registraron por separado la evidencia local completa y la evidencia remota del bloque 1.
5. **Higiene del repositorio.** Se restauraron reglas para excluir credenciales, tokens, `kaggle.json`,
   entornos y salidas temporales.
6. **Entrega final.** Se generaron un reporte editable en DOCX y una presentación editable en PPTX,
   junto con sus PDFs, usando únicamente resultados medidos.

## Evidencia en Kaggle

El bloque 1 sí cuenta con ejecución privada en Kaggle CPU: notebook versión 1, estado Successful,
runtime reportado de 28 s y controles coincidentes con la ejecución local. El dataset privado v1 es
`sarahvasquez97/ksvd-dani-datos-v1`. La auditoría no afirma que el pipeline completo de los cuatro
bloques se haya ejecutado en Kaggle; esa parte quedó verificada localmente.

## Limitaciones que permanecen

El experimento usa una sola imagen de prueba, una sola realización de ruido y tres valores prefijados
de dispersión. No hay validación externa ni comparación con el estado del arte. Los tiempos dependen de
la máquina. Estas limitaciones no invalidan la prueba de concepto, pero impiden generalizar el resultado.
