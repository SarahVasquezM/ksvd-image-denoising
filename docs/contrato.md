# Contrato de integración 1

Se aplica el plan general junto con el alcance actual de Dani. No se detectaron
contradicciones que requieran cambios en firmas o formatos. El documento de Dani
precisa el umbral de elegibilidad: norma centrada >1e-8. El código lo conserva.

```python
extract_patches(image, patch_size=8, stride=4)  # Z, means, positions
assemble_patches(patches, positions, image_shape, patch_size=8)  # image
```

Z=(p*p,N) float64, means=(N,) float64, positions=(N,2) enteras. Recorrido por
filas/columnas, aplanado C. Añadir último inicio válido por eje sin duplicarlo.
Se validan enteros positivos, imagen 2D real finita y lado >= p. Para garantir
cobertura se rechaza stride > patch_size. No hay padding ni normalización.
Ensamblado exige posiciones enteras dentro de la imagen y cobertura positiva;
recibe medias YA sumadas y promedia solapamientos sin clipping.

Paquete: datos.npz con train_image=(256,256), test_image=(128,128),
X_train=(64,1500) float64 y train_indices=(1500,) enteros sobre la extracción
completa. Configuración JSON separada, sin pickle. Incluye todos los parámetros
de los bloques posteriores sin ejecutarlos. El resize explicita sus valores
predeterminados para evitar ambigüedad entre entornos.

Las imágenes no se normalizan por su máximo; img_as_float64 convierte la escala
entera antes del resize. El parámetro clip de resize conserva el rango de entrada;
extract_patches y assemble_patches no recortan valores, incluidas futuras entradas
ruidosas fuera de [0,1].

Fundamento consultado: artículo adjunto, pp. 3736–3738, representación de parches
por columnas y agregación de solapamientos. Nuestro promedio simple no implementa
la relajación con la imagen ruidosa del estimador completo del artículo.
