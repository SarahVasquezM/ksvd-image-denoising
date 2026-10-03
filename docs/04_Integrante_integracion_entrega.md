# Integrante 4 — Integración, figuras, reporte y exposición

## Tu responsabilidad y plazo

Mantener el notebook final y convertir los paquetes técnicos en una entrega verificable: modelo, resultados, reporte de máximo 5 páginas y presentación de máximo 15 minutos. Empiezas el día 1 a las 09:15 con estructura y fundamentos. Recibes datos a las 12:00, modelo a las 17:00 y resultados el día 2 a las 12:00. Entrega final: día 2 a las 17:00; ensayo común hasta las 18:00.

Carga estimada: 5–6 horas efectivas más lectura/ensayo. No tienes que inventar ni redactar todo: Dani entrega datos/objetivo, persona 2 método y persona 3 resultados/limitaciones. Integra esos textos, conserva su sentido y devuelve únicamente errores críticos; las correcciones de estilo son tuyas.

## Herramientas

Kaggle CPU, Python, NumPy, pandas y Matplotlib. Editor de texto/Word/LaTeX para reporte; PowerPoint/Google Slides o similar para presentación. Elegir lo que ya dominen. No aprender una herramienta nueva durante estos dos días.

## Trabajo paralelo del día 1

1. Crear `04_final` con secciones y rutas configurables; no copiar cada notebook entero.
2. Preparar lectura de los módulos `.py` de cada paquete. Los notebooks técnicos documentan el desarrollo, pero los módulos compartidos evitan copias divergentes.
3. Crear ocho diapositivas vacías y estructura de reporte con presupuesto de páginas.
4. Redactar fundamento a partir de los dos papers, con citas. Explicar que es una adaptación pequeña, no una réplica completa.
5. Preparar funciones de figuras con arrays/tablas de ejemplo claramente marcados. No exportarlas como resultados.
6. Registrar la versión aceptada de cada paquete recibido y verificar la existencia de sus archivos.

## Notebook final

Orden de 10 secciones:
1. Objetivo, alcance y referencias.
2. Entorno, versiones, rutas y configuración efectiva.
3. Carga de imágenes y demostración de parches.
4. Explicación e importación de OMP/K-SVD.
5. Entrenamiento opcional o carga del diccionario ya guardado.
6. Visualización del diccionario e historial.
7. Generación de ruido y reconstrucciones.
8. Métricas y tabla.
9. Tres figuras finales.
10. Conclusiones, limitaciones e instrucciones de reproducción.

Usar `RETRAIN=False` por defecto para la demostración, pero dejar una ruta ejecutable con `RETRAIN=True` que reproduce entrenamiento. Mostrar qué modo se ejecutó. No decir que el notebook entrenó si sólo cargó un archivo.

El paquete final debe incluir modelo.py, preprocesamiento.py, evaluacion.py, datos, modelo, configuración, historial, resultados, imágenes reconstruidas y versiones. Ejecutar desde una sesión limpia permite detectar dependencias de variables creadas fuera de orden.

## Tres figuras definitivas

1. Panel de cuatro imágenes, T0=4: original; ruidosa; reconstrucción de entrada limpia; reconstrucción de entrada ruidosa. Misma escala visual vmin=0,vmax=1. Rótulos con métricas reales. La cuarta imagen es la demostración de denoising.
2. Mosaico del diccionario entrenado: cada columna de D se muestra como 8×8. Normalizar sólo para visualizar átomos; no modificar D. Ajustar cuadrícula al K efectivo y ocultar celdas vacías.
3. PSNR frente a máximo de átomos utilizados POR PARCHE (2,4,8), con curvas para entrada limpia y ruidosa y línea horizontal del ruido sin procesar. No llamar a este eje «número de componentes del diccionario». No imponer monotonicidad.

Exportar PNG con texto legible, por ejemplo 150–200 dpi. Una tabla compacta presenta MSE, PSNR, actividad media y tiempo. No usar resultados del paper como si fueran propios.

Si el experimento se redujo a T0=4, reemplazar figura 3 por una comparación de los casos disponibles; no dibujar puntos inexistentes.

## Reporte: máximo cinco páginas, referencias incluidas

- 0.75 páginas: objetivo/fundamento. Integra objetivo de Dani y referencias de los papers.
- 0.75: datos y preparación, desde nota_datos.md.
- 1.5: proceso y configuración efectiva, desde nota_metodo.md.
- 1.5: resultados y análisis, desde nota_resultados.md.
- 0.5: conclusiones y referencias.

Usar tres figuras compactas o reducir alguna si el texto queda ilegible. No incluir listado completo de código. Debe quedar explícito dónde entra la SVD, qué aprende el diccionario, cómo se calculan coeficientes y cómo se compara la salida. Si se activó contingencia, actualizar todas las cifras y dimensiones.

## Presentación y reparto

8 diapositivas; objetivo de ensayo 13 minutos:
- Dani, diapositivas 1–2, 3 min: objetivo, datos, parches.
- Persona 2, 3–4, 4 min: representación dispersa, OMP, actualización K-SVD.
- Persona 3, 5–6, 3 min: experimentos y resultados.
- Persona 4, 7–8, 3 min: interpretación, limitaciones y cierre.

La demo en vivo, si la hay, sustituye parte de esos 13 minutos; no se añade encima. Llevar capturas/resultados guardados. Nunca entrenar durante la exposición.

## Cronograma final y aceptación

Día 2, 12:00–15:30: integrar resultados, generar figuras, completar redacción y presentación.
15:30–16:30: reiniciar entorno, cargar paquetes y ejecutar la ruta completa de inferencia; comprobar que tablas coinciden con arrays y que la ruta de reentrenamiento tiene código e instrucciones.
16:30–17:00: exportar, contar páginas y entregar copia descargada.
17:00–18:00: ensayo con los cuatro. Corregir afirmaciones o etiquetas, no añadir funciones.

Lista final:
- [ ] K-SVD se ejecutó realmente y hay diccionario e historial.
- [ ] Ninguna celda ficticia o dato de desarrollo se presenta como resultado.
- [ ] Imágenes y métricas provienen de las mismas ejecuciones.
- [ ] Se distinguen limpieza de ruido y reconstrucción de imagen limpia.
- [ ] Se declaran dos imágenes totales, una de prueba, una realización de ruido y ausencia de validación/generalización.
- [ ] Se identifican NumPy/SVD y scikit-learn/OMP como herramientas reutilizadas.
- [ ] Reporte PDF de ≤5 páginas; exposición cronometrada de ≤15 minutos.
- [ ] Notebook, módulos y resultados están descargados además de guardados en Kaggle.

Eres responsable de la integración final; no significa reprogramar todo. Ante un defecto crítico del núcleo, comunica evidencia concreta al autor y usa el margen. Ante una mejora estética opcional, prioriza entregar.

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
