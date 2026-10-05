# Comparación de las dos rutas K-SVD

Se evaluaron dos estrategias con la misma realización de ruido, valores de T0, métricas y ensamblado: un diccionario externo aprendido con `camera` limpia y un diccionario adaptativo aprendido directamente de `coins` ruidosa. La segunda estrategia sigue la idea de la Sec. III-B de Elad y Aharon (2006); la referencia limpia no participa en su entrenamiento.

## Mejores resultados sobre la entrada ruidosa

| diccionario | T0 | MSE | PSNR (dB) |
|---|---:|---:|---:|
| adaptativo_coins_ruidosa | 2 | 0.002459 | 26.09 |
| externo_camera_limpia | 4 | 0.002442 | 26.12 |

## Alcance

El diccionario adaptativo usó 961 parches de la misma observación ruidosa, K=128 y 8 iteraciones. La comparación es transductiva: adaptar el diccionario a la entrada es parte del método y no equivale a validación sobre una imagen no vista. Se conservó el protocolo reducido del proyecto para aislar el origen del diccionario; `config_adaptativa.json` enumera las diferencias frente al estimador completo del artículo.

## Productos

- `externo/`: evaluación nueva del modelo de `02_modelo/` bajo la misma ejecución de tiempos.
- `adaptativo/`: modelo, historial, configuración, evaluación y reconstrucciones de la ruta adaptativa.
- `comparacion_resultados.csv`: referencia única y los seis casos de cada diccionario.
- `verificacion_comparacion.json`: comprobaciones de ruido, T0, forma y normas del diccionario.
- `figuras/`: diccionarios, curvas de PSNR y reconstrucciones comparables.
