# Metodología

## Fundamento
* **K-SVD** (Aharon, Elad y Bruckstein, 2006, doi:10.1109/TSP.2006.881199): aprende un diccionario
  alternando codificación dispersa de los ejemplos con el diccionario actual y actualización de los
  átomos uno por uno, junto con los coeficientes que los usan, mediante la SVD de un residual.
* **Denoising con diccionarios aprendidos** (Elad y Aharon, 2006, doi:10.1109/TIP.2006.881969).
  Del artículo (verificado en el PDF del autor): parches 8×8 con solapamiento; diccionarios de
  64×256; DCT redundante como diccionario inicial/de referencia; OMP que acumula átomos hasta que el
  error baja de un umbral `(Cσ)²` con `C = 1.15`; `J = 10` iteraciones; promedio final con la imagen
  ruidosa ponderado por `λ = 30/σ`; la actualización del diccionario aplica la SVD a residuales
  "computed only on the examples that use this atom"; los resultados de su Tabla I promedian cinco
  experimentos.

## Qué se adapta y qué se simplifica
| Aspecto | Elad y Aharon (2006) | Este proyecto |
|---|---|---|
| Entrenamiento | base de parches limpios o la propia imagen ruidosa | 1500 parches de *camera* limpia |
| Diccionario | 64 × 256, inicio DCT | 64 × 128, inicio con parches de datos |
| Codificación | OMP con umbral de error | OMP con número fijo T0 ∈ {2,4,8} |
| Reconstrucción | promedio de parches + imagen ruidosa (λ) | sólo promedio de parches |
| Evaluación | varias imágenes, σ y 5 realizaciones | una imagen, σ = 20/255, una realización |

## Diseño experimental
1. Datos (bloque 1): ver `01_datos/nota_datos.md`.
2. Modelo (bloque 2): ver `02_modelo/nota_metodo.md`.
3. Evaluación (bloque 3): ver `03_evaluacion/nota_resultados.md`.
4. Integración (bloque 4): verificación cruzada y reporte; ver `04_final/README_integracion.md`.

## Verificación
* Pruebas unitarias y de integración en `tests/` (contratos, casos borde, prueba de rango 1 de la
  SVD, reproducción completa del pipeline en un directorio temporal).
* Comprobaciones obligatorias dentro del propio código (ida y vuelta < 1e-10, métricas de control,
  validación del modelo, coherencia entre bloques); cualquier fallo detiene la ejecución.
* Auditoría final desde un entorno limpio (ver `reproduccion.md`).

## Principios de reporte
* Ninguna cifra se escribe antes de ejecutar el experimento; reporte y presentación leen los números
  de los archivos de resultados.
* Las interpretaciones se marcan como tales; no se generaliza desde una realización de ruido.
