# Integrante 2 — OMP, K-SVD y diccionario entrenado

## Tu responsabilidad y plazo

Entregar una implementación pequeña de K-SVD real, comprobada, y su diccionario entrenado. Empiezas día 1 a las 09:15 con datos sintéticos, recibes datos reales a las 12:00 y entregas a las 17:00. Carga estimada: 5–6 horas más lectura/ensayo.

No implementes OMP ni algoritmos numéricos de SVD desde cero. Reutiliza `sklearn.linear_model.orthogonal_mp` y `numpy.linalg.svd`. No sustituyas el aprendizaje por DictionaryLearning.

## Herramientas y entradas

Kaggle CPU, NumPy, scikit-learn, pandas o csv, time y json. Desde las 09:15 usa una matriz sintética (64,200), centrada por columna, sólo para comprobar el código. A las 12:00 cambia a X_train del paquete de Dani sin modificar firmas ni dimensiones de entrada.

## Funciones públicas

1. `sparse_code(D, Z, max_nonzero) -> A` con D=(64,K), Z=(64,N), A=(K,N).
2. `fit_ksvd(X_train, n_atoms=128, n_iter=8, train_sparsity=4, seed=43) -> (D, A_train, history)`.

Para OMP, el argumento X de la función de sklearn es nuestro diccionario D; su argumento y es nuestro Z. Utiliza columnas de D normalizadas. La función sin estimador evita añadir un intercepto; las señales ya están centradas. Trata columnas de señal casi nulas devolviendo ceros y conserva A bidimensional cuando N=1.

No ocultes todas las advertencias de dependencia lineal: registra las relevantes y comprueba finitud/reconstrucción. Si se seleccionan menos átomos por degeneración, max_nonzero sigue siendo un máximo.

## Núcleo matemático

Con X de forma (64,N), A de forma (K,N), para cada átomo j:

- omega = índices de columnas i con A[j,i] distinto de cero.
- E = X[:,omega] - D @ A[:,omega] + outer(D[:,j], A[j,omega]).
- U,s,Vt = np.linalg.svd(E, full_matrices=False).
- D[:,j] = U[:,0].
- A[j,omega] = s[0] * Vt[0,:].

El residual se restringe a los parches que usan el átomo. Aplicar SVD al residual completo sin esa selección cambia el método y puede destruir la dispersión. Usa el estado actualizado de D y A en cada paso.

Si omega está vacío, reinicializa el átomo con un residual de entrenamiento no nulo normalizado; puede permanecer sin uso hasta la siguiente codificación. Si todos los residuales son casi nulos, conserva el átomo. Inicializa átomos con columnas no constantes de X normalizadas, usando RNG 43. El centrado de los datos debe conservarse en las actualizaciones salvo redondeo.

Después de la última iteración vuelve a codificar X con el D final para que A_train y el error final sean consistentes. No prometas descenso estricto del error en todo el entrenamiento: OMP es aproximado y existen reinicializaciones. Para una actualización SVD de soporte fijo, la aproximación de rango uno sí debe ser coherente con el residual.

## Plan por horas

- 09:15–11:30: escribir sparse_code y fit_ksvd; comprobar dimensiones, señal cero y soporte limitado con una entrada pequeña.
- 11:30–12:00: preparar guardado de modelo e historial.
- 12:00–13:00: cargar datos de Dani y ejecutar 1–2 iteraciones piloto; medir tiempo real.
- 13:00–15:30: entrenar 128 átomos, 1,500 parches, 8 iteraciones, sparsity=4.
- 14:30: decidir si hace falta contingencia, según tiempo medido. No asumir entrenamiento de «unos minutos» sin medir.
- 15:30–16:30: validar modelo y escribir método.
- 16:30–17:00: guardar y entregar paquete único.

## Contingencia

Primero reducir a 500 parches/5 iteraciones conservando 128 átomos. Si aún no alcanza, usar 64 átomos. Los consumidores obtienen K de D.shape[1]. Registrar la configuración efectiva y que un diccionario 64×64 no es sobrecompleto. Mantener el barrido de evaluación 2/4/8.

## Aceptación y entregables

- [ ] D tiene forma (64,K), valores finitos y columnas de norma cercana a 1.
- [ ] A_train corresponde al D final y tiene forma (K,N_train).
- [ ] Cada columna de A_train tiene como máximo 4 entradas >1e-10 en valor absoluto.
- [ ] Reconstrucción X_hat=D@A_train tiene forma correcta y error finito.
- [ ] La actualización SVD se verifica en un ejemplo pequeño contra el error de rango uno antes de actualizar.
- [ ] Diccionario e historial se pueden cargar sin volver a entrenar.

Entregar `modelo.py`, `02_modelo.ipynb`, `modelo.npz` (D,D_init,A_train), `historial.csv`, `config_efectiva.json` y `nota_metodo.md` (350–450 palabras, fórmulas, inicialización, parada, papel de las librerías, tiempo real y limitaciones).

El historial lleva iteration, train_mse, elapsed_s, unused_atoms. Documenta si el error se mide después de recodificar o actualizar; sé consistente. Config efectiva contiene toda la configuración base y cualquier reducción aplicada.

## Después de entregar

Tu bloque técnico termina a las 17:00 del día 1. La persona 3 aplica tu modelo y la 4 integra. Tú preparas la explicación de 4 minutos: OMP encuentra pesos; K-SVD cambia átomos; la SVD del residual produce una actualización de rango uno. No digas que SVD global, PCA, eigenfaces o DictionaryLearning son K-SVD.

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
