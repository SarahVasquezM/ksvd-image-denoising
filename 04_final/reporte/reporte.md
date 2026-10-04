# Reconstrucción y eliminación de ruido con K-SVD — reporte

> Versión Markdown generada automáticamente por `tools/build_report.py` a partir de los archivos de resultados.
> La versión con formato es `reporte.pdf`.

## Configuración efectiva
K = 128 (sobrecompleto), 1500 parches,
8 iteraciones, T = 4, nivel de contingencia 0 (configuración deseada).
Tiempo de entrenamiento: 2.09 s. MSE de entrenamiento: 9.549e-04 (D_init) → 3.413e-04 (D final).

## Resultados (coins 128×128, σ = 20/255, una realización, RNG 44)
| entrada | T0 | MSE | PSNR (dB) | coef. activos | tiempo (ms) |
|---|---:|---:|---:|---:|---:|
| ruidosa (sin procesar) | -- | 0.006179 | 22.09 | -- | -- |
| limpia | 2 | 0.002132 | 26.71 | 2.00 | 52.2 |
| limpia | 4 | 0.001264 | 28.98 | 4.00 | 88.5 |
| limpia | 8 | 0.000719 | 31.43 | 8.00 | 179.4 |
| ruidosa | 2 | 0.002784 | 25.55 | 2.00 | 51.9 |
| ruidosa | 4 | 0.002442 | 26.12 | 4.00 | 94.0 |
| ruidosa | 8 | 0.002730 | 25.64 | 8.00 | 168.1 |

## Conclusiones
- Con la imagen limpia, el PSNR crece monótonamente con T0 (de 26.71 dB con T0=2 a 31.43 dB con T0=8): más átomos representan mejor la señal.
- Con la imagen ruidosa el PSNR no es monótono: el máximo está en T0=4 (26.12 dB). Esto es consistente con un compromiso: con pocos átomos se pierde detalle y con muchos OMP también ajusta parte del ruido (interpretación, no medida directa).
- Frente a la imagen ruidosa sin procesar (22.09 dB), la mejor reconstrucción mejora 4.03 dB. Es una ganancia real pero modesta; no se compara con el estado del arte.
- La actividad dispersa media coincide exactamente con T0; el tiempo de reconstrucción crece con T0 y la codificación OMP ocupa en promedio el 88% de ese tiempo.

![resumen](../figuras/resumen_resultados.png)
