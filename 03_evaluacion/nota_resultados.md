# Nota del bloque 3 — Evaluación y resultados

**Responsable original:** integrante 3. **Implementación:** `src/evaluacion.py` (fuente única;
`03_evaluacion/evaluacion.py` sólo la reexporta). **Notebook:** `03_experimentos.ipynb`.

## Protocolo
`run_experiments(data_path, model_path, config_path, output_dir)`:

1. Lee `datos.npz`, `modelo.npz` y `config.json`. `K`, la dimensión de átomo, `patch_size`, `stride`,
   `sigma`, semillas y la lista de T0 salen de esos archivos (no se codifican a mano).
2. **Comprobación previa de métricas** (falla si no se cumple):
   imagen contra sí misma → MSE = 0, PSNR = ∞; diferencia constante 0.1 → MSE = 0.01, PSNR = 20 dB.
   Resultado obtenido: `0.0`, `inf`, `0.01`, `20.0`.
3. **Ruido:** `rng = np.random.default_rng(44)`; `noise = rng.normal(0, 20/255, size=test_image.shape)`;
   `noisy = test_image + noise`. **Una sola realización**, reutilizada para T0 = 2, 4, 8. Sin
   *clipping* (modelo aditivo gaussiano puro). σ empírica: 0.07861 (nominal 0.07843).
4. Para cada condición (limpia, ruidosa) y cada T0: extraer parches centrados de la imagen de
   **entrada** → `sparse_code(D, Z, T0)` → `D @ A + means` → `assemble_patches` (promedio de
   solapamientos, sin clipping).
5. Métricas **siempre contra la imagen limpia**: `skimage.metrics.mean_squared_error` y
   `peak_signal_noise_ratio(data_range=1.0)`; número medio (y máximo) de coeficientes con |a| > 1e-10;
   tiempo = mediana de 5 repeticiones idénticas de extracción + OMP + reensamblado (también se
   guarda el tiempo sólo de OMP). Se verifica que las 5 repeticiones den la misma imagen.

## Resultados (coins 128 × 128, 961 parches, K = 128)

| entrada | T0 | MSE | PSNR (dB) | coef. activos medios |
|---|---:|---:|---:|---:|
| ruidosa sin procesar | — | 0.006179 | 22.09 | — |
| limpia | 2 | 0.002132 | 26.71 | 2.00 |
| limpia | 4 | 0.001264 | 28.98 | 4.00 |
| limpia | 8 | 0.000719 | 31.43 | 8.00 |
| ruidosa | 2 | 0.002784 | 25.55 | 2.00 |
| ruidosa | 4 | 0.002442 | 26.12 | 4.00 |
| ruidosa | 8 | 0.002730 | 25.64 | 8.00 |

MSE y PSNR son deterministas (se reproducen exactamente en `tests/test_integracion.py`). Los tiempos
dependen de la máquina y la corrida; los de la última ejecución están en `resultados.csv`
(del orden de 45 ms con T0 = 2, 85 ms con T0 = 4 y 155 ms con T0 = 8, con 4 hilos BLAS).

## Lectura
* **Imagen limpia:** el PSNR aumenta con T0 (26.71 → 28.98 → 31.43 dB). Es la capacidad de
  representación del diccionario sobre una imagen no vista: más átomos, menor error.
* **Imagen ruidosa:** el PSNR no es monótono: 25.55 (T0=2) → **26.12 (T0=4)** → 25.64 dB (T0=8).
  El mejor caso mejora **4.03 dB** frente a la entrada ruidosa (22.09 dB). La forma de la curva es
  consistente con el compromiso esperado (pocos átomos pierden detalle; muchos átomos también
  ajustan ruido), pero esta interpretación no se midió directamente.
* **Actividad dispersa:** la media coincide exactamente con T0 en todos los casos (ningún parche de
  prueba agotó su residual antes de T0).
* **Tiempo:** crece aproximadamente lineal con T0; la codificación OMP es la mayor parte (≈ 90 %).

### Control: `D_init` frente a `D` aprendido (análisis complementario del notebook)
Mismo protocolo cambiando sólo el diccionario (PSNR en dB):

| T0 | limpia D_init | limpia D | ruidosa D_init | ruidosa D |
|---:|---:|---:|---:|---:|
| 2 | 26.36 | 26.71 | 25.34 | 25.55 |
| 4 | 28.39 | 28.98 | 25.93 | 26.12 |
| 8 | 30.83 | 31.43 | 25.59 | 25.64 |

El aprendizaje K-SVD mejora en todos los casos, pero la ganancia es pequeña (≈ 0.35–0.6 dB en la
limpia, ≈ 0.05–0.2 dB en la ruidosa): con sólo 1500 parches y 8 iteraciones, buena parte del
desempeño se debe ya a usar parches naturales como átomos.

## Productos
| Archivo | Contenido |
|---|---|
| `resultados.csv` | `condition, T0, mse, psnr_db, mean_active_coefs, max_active_coefs, reconstruction_time_s, coding_time_s, n_patches, n_atoms` |
| `reconstrucciones.npz` | `test_image`, `noisy_image`, `noise`, `rec_limpia_T{2,4,8}`, `rec_ruidosa_T{2,4,8}` |
| `evaluacion_meta.json` | comprobaciones de métricas, σ empírica, número de realizaciones, protocolo de tiempo |
| `figuras/` | reconstrucciones, mapas de error, PSNR y MSE vs T0, actividad y tiempo |

## Limitaciones
* Una imagen de prueba y **una** realización de ruido: no hay barras de error ni prueba estadística;
  diferencias de décimas de dB entre T0 no deben generalizarse.
* T0 fijo en lugar del criterio de error residual usado por Elad y Aharon (2006); sin el promedio
  ponderado con la imagen ruidosa; diccionario entrenado en otra imagen limpia, no en la ruidosa.
* Las métricas se calculan sobre la reconstrucción sin clipping (puede salir ligeramente de [0, 1]).
* No se compara con otros métodos de eliminación de ruido ni se afirma superar el estado del arte.
