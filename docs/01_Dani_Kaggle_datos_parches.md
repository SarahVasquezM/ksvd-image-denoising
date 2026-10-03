# Integrante 1 — Dani: entorno, datos y parches

## Tu responsabilidad y plazo

Preparar el entorno reproducible, entregar imágenes y parches correctos, y dejar listas las funciones de extracción y reconstrucción. Tu bloque técnico termina el día 1 a las 12:00. No te corresponde volver después a integrar entrenamiento, métricas o reporte final.

Carga estimada: 4–5 horas, incluyendo preparación, comprobaciones y documentación, además de lectura y ensayo común. El alcance adicional para equilibrar tu parte es la función de ensamblado y la guía de reproducción del entorno.

## Herramientas

Kaggle con CPU; Python; NumPy; scikit-image; Matplotlib; módulos estándar json y pathlib. Comprobar que scikit-learn puede importarse para el siguiente integrante. No actualizar todos los paquetes si ya funcionan. Registrar versiones instaladas; instalar sólo si falta una dependencia y registrar el cambio.

## Pasos de trabajo

1. Crear `01_datos` en Kaggle y acordar cómo compartir notebooks/paquetes privados. Los demás pueden crear los suyos desde el comienzo; no necesitan esperar tu notebook.
2. Ejecutar importaciones y registrar Python y versiones de numpy, scipy, sklearn, skimage y matplotlib en `versiones.txt`.
3. Crear carpeta de salida `/kaggle/working/entrega_01/`. Escribir archivos allí, no en las entradas montadas.
4. Cargar `data.camera()` y `data.coins()`. Ya son grises. Convertir con img_as_float64 antes del redimensionamiento.
5. Redimensionar camera a 256×256 y coins a 128×128 con anti_aliasing=True y preserve_range=True. Conservar ambas imágenes y registrar sus nombres, formas y rango.
6. Programar `extract_patches` y `assemble_patches` con las firmas exactas del plan general.
7. Extraer parches 8×8, paso 4, centrados, con sus medias y posiciones. No normalizar la energía de cada parche.
8. Filtrar parches de entrenamiento con norma centrada >1e-8. Muestrear 1,500 sin reemplazo con default_rng(42). Conservar índices respecto de la extracción completa.
9. Guardar `datos.npz` y `config.json` según el contrato.
10. Generar una figura de las dos imágenes y otra pequeña de parches para inspección. Explicar cuáles son datos de entrenamiento y cuáles de prueba.
11. Escribir `nota_datos.md`: procedencia, preparación, separación por imagen, tamaño de parche, muestreo, semillas, comprobaciones y limitaciones. Incluir propuesta de objetivo en un párrafo. Extensión: 250–350 palabras más instrucciones breves para reproducir.
12. Guardar versión con salidas y compartir paquete único a personas 2, 3 y 4 antes de las 12:00.

## Contrato que debes entregar

`preprocesamiento.py`:
- `extract_patches(image, patch_size=8, stride=4)` devuelve Z=(64,N), means=(N,), positions=(N,2).
- `assemble_patches(patches, positions, image_shape, patch_size=8)` recibe parches SIN centrar, ya con sus medias restauradas, y devuelve imagen por promedio de solapamientos.
- Aplanado C, recorrido por filas/columnas; no hacer clipping en ninguna de estas funciones.

`datos.npz`:
- train_image=(256,256), test_image=(128,128), X_train=(64,1500), train_indices=(1500,).
- Imágenes/parches float64; índices enteros. El JSON es configuración, no se guarda como un objeto pickle dentro del NPZ.

## Comprobaciones antes de entregar

- [ ] Imágenes en [0,1], sin NaN ni infinito.
- [ ] train y test provienen de imágenes diferentes.
- [ ] Cada columna de X_train tiene media aproximadamente cero; no hay columnas constantes.
- [ ] `assemble_patches(Z+means[None,:],positions,image.shape)` recupera train y test con error máximo <1e-10.
- [ ] El mapa de cobertura no tiene ceros, incluidas esquinas y bordes.
- [ ] Se pueden cargar los archivos en una sesión nueva.
- [ ] El paquete incluye código ejecutable, no sólo las imágenes o capturas.

## Trabajo paralelo y cierre

A las 09:15 confirma únicamente que mantienes las firmas de este plan. La persona 2 ya puede usar X sintético de forma (64,200), y la 3 prepara evaluación sin tus datos reales. No cambies las formas durante tu bloque.

Entrega: `01_datos.ipynb`, `preprocesamiento.py`, `datos.npz`, `config.json`, `versiones.txt`, `nota_datos.md` y figuras de inspección. Pide confirmación breve de carga a las 12:00. Una vez aceptado, tu desarrollo termina; prepara tu explicación de 3 minutos para el ensayo del día 2.

## Debes poder explicar

Por qué dividir imágenes en parches; cómo un parche se transforma en vector; por qué centrarlo y devolver su media; cómo el promedio resuelve las zonas superpuestas; por qué no se usó la imagen de prueba para aprender el diccionario.

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
