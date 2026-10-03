# Presentación — guion

Generado por `tools/build_report.py`; diapositivas en `presentacion.pdf`.


## 1. Reconstrucción y eliminación de ruido con K-SVD

- Adaptación didáctica pequeña de K-SVD (Aharon, Elad y Bruckstein, 2006; Elad y Aharon, 2006).
- Diccionario aprendido sobre parches 8×8 de camera; evaluación en coins (no vista).
- Proyecto de cuatro bloques: datos · modelo · evaluación · integración.
- Matemáticas Computacionales — INAOE

## 2. Bloque 1 — Datos y parches

- camera → 256×256 (entrenamiento), coins → 128×128 (prueba).
- Parches 8×8, paso 4, centrados: 3969 (camera), 961 (coins).
- 1500 parches de entrenamiento sin reemplazo (RNG 42).
- Ida y vuelta extraer/reensamblar: error máx 1.1e-16.

![](..\..\01_datos\figuras\muestra_parches.png)

## 3. Bloque 2 — K-SVD propio

- OMP: sklearn.linear_model.orthogonal_mp (≤ T átomos).
- Átomo j: SVD del residual restringido E_j = X_ω − D A_ω + d_j a_j; d_j ← u₁, a_j ← σ₁v₁ᵀ.
- Átomos sin uso → residual de mayor norma, normalizado.
- K=128, 1500 parches, 8 iteraciones, T=4; 2.09 s en CPU; sin contingencia.

![](..\figuras\diccionarios.png)

## 4. Bloque 2 — Convergencia

- MSE de entrenamiento: 9.55e-04 (D_init) → 3.41e-04 (D final).
- No se garantiza descenso estricto entre iteraciones (OMP es aproximado); sí dentro de cada actualización de átomo.

![](..\figuras\curva_entrenamiento.png)

## 5. Bloque 3 — Resultados


| entrada | T0 | MSE | PSNR (dB) | coef. activos | tiempo (ms) |
|---|---|---|---|---|---|
| ruidosa (sin procesar) | -- | 0.006179 | 22.09 | -- | -- |
| limpia | 2 | 0.002132 | 26.71 | 2.00 | 52.2 |
| limpia | 4 | 0.001264 | 28.98 | 4.00 | 88.5 |
| limpia | 8 | 0.000719 | 31.43 | 8.00 | 179.4 |
| ruidosa | 2 | 0.002784 | 25.55 | 2.00 | 51.9 |
| ruidosa | 4 | 0.002442 | 26.12 | 4.00 | 94.0 |
| ruidosa | 8 | 0.002730 | 25.64 | 8.00 | 168.1 |

## 6. Bloque 3 — Reconstrucciones


![](..\figuras\reconstrucciones.png)

## 7. Efecto de la dispersión T0

- Con la imagen limpia, el PSNR crece monótonamente con T0 (de 26.71 dB con T0=2 a 31.43 dB con T0=8): más átomos representan mejor la señal.
- Con la imagen ruidosa el PSNR no es monótono: el máximo está en T0=4 (26.12 dB). Esto es consistente con un compromiso: con pocos átomos se pierde detalle y con muchos OMP también ajusta parte del ruido (interpretación, no medida directa).

![](..\figuras\psnr_vs_t0.png)

## 8. Conclusiones y limitaciones

- Frente a la imagen ruidosa sin procesar (22.09 dB), la mejor reconstrucción mejora 4.03 dB. Es una ganancia real pero modesta; no se compara con el estado del arte.
- La actividad dispersa media coincide exactamente con T0; el tiempo de reconstrucción crece con T0 y la codificación OMP ocupa en promedio el 88% de ese tiempo.
- Limitaciones: una imagen de prueba, una realización de ruido, T0 fijo (no criterio de error), sin promedio ponderado con la imagen ruidosa; no se afirma superar el estado del arte.
