# Procedencia y referencias verificadas

La API oficial de [skimage.data](https://scikit-image.org/docs/stable/api/skimage.data.html)
identifica `camera()` como una imagen uint8 de 512×512, liberada por el fotógrafo
Lav Varshney bajo CC0. Identifica `coins()` como monedas griegas de Pompeya,
uint8 de 303×384, procedentes de Brooklyn Museum Collection, sin restricciones
de copyright conocidas. No se convierte esta última descripción en una licencia
CC0 inventada ni se extiende una licencia de imagen al código del proyecto.

Se redimensionaron ambas imágenes y se conservaron por separado entrenamiento y
prueba. Las figuras del paquete son inspecciones de estos datos y sus parches.
No son resultados aprendidos. Si Kaggle pide una licencia para el conjunto mixto,
usar una opción sin licencia especificada si está disponible; no asignar una
licencia de redistribución que el equipo no haya acordado y verificado.

El PDF adjunto de Elad y Aharon se consultó localmente (páginas 3736–3738).
Describe representaciones por parches y agregación de solapamientos. La ecuación
completa incluye una relación con la imagen ruidosa; el promedio simple del
contrato del equipo es una adaptación didáctica. El archivo del artículo no se
incluyó en Git ni en el paquete de imágenes.

- [Artículo TIP, DOI 10.1109/TIP.2006.881969](https://doi.org/10.1109/TIP.2006.881969).
- [Artículo K-SVD, DOI 10.1109/TSP.2006.881199](https://doi.org/10.1109/TSP.2006.881199): referencia suministrada por el plan; no se atribuyen resultados numéricos.
- [resize oficial](https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize).

La preparación no demuestra mejoras de PSNR ni generalización a otras imágenes.
