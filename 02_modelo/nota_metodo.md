# Nota del bloque 2 — OMP y K-SVD

**Responsable original:** integrante 2. **Implementación:** `src/modelo.py` (fuente única;
`02_modelo/modelo.py` sólo la reexporta). **Notebook:** `02_modelo.ipynb`.

## Representación dispersa
Cada parche centrado `x ∈ R^64` se aproxima como `x ≈ D a`, donde `D ∈ R^{64×K}` es un diccionario
con columnas (átomos) de norma 1 y `a ∈ R^K` tiene a lo más `T` entradas no nulas. Con `K = 128 > 64`
el diccionario es **sobrecompleto** (redundante): hay muchas formas de representar un parche y se
busca la más dispersa.

## Función de OMP — `sparse_code(D, Z, max_nonzero) -> A`
* Llama a `sklearn.linear_model.orthogonal_mp(D, Z, n_nonzero_coefs=max_nonzero, precompute=True)`.
  OMP elige en cada paso el átomo más correlacionado con el residual y re-ajusta por mínimos
  cuadrados todos los coeficientes elegidos. **No se reimplementa OMP.**
* Sin intercepto (los parches ya están centrados).
* Exige `D` con columnas de norma 1 (error si no) y entradas finitas.
* Señales con norma ≤ 1e-8 → columna de ceros, sin llamar a OMP.
* `A` siempre es (K, N), también si N = 1 o si se pasa un vector.
* `max_nonzero` es un **máximo**: si el residual se anula antes, OMP se detiene. scikit-learn emite
  un `RuntimeWarning` por señal en ese caso; `sparse_code` los registra y re-emite **una**
  advertencia resumida con el conteo. Cualquier otra advertencia se re-emite intacta (no se suprime
  nada indiscriminadamente).
  En la iteración 1 aparecen exactamente 128 de estas advertencias: son los 128 parches que se
  usaron como átomos iniciales, que se representan con un solo átomo y residual cero. Es esperado.

## Función del diccionario
Los átomos son "parches prototipo" (bordes, gradientes, texturas). Un diccionario aprendido se
adapta a la estadística local de las imágenes y permite representaciones más dispersas y precisas
que el mismo número de parches elegidos al azar (ver `figuras/diccionario_inicial_vs_final.png`).

## Actualización K-SVD — `ksvd_dictionary_update(X, D, A)`
Para cada átomo `j = 0..K−1`, **usando D y A ya actualizados por los átomos anteriores**:

```
omega = índices i donde A[j, i] != 0
E     = X[:, omega] - D @ A[:, omega] + outer(D[:, j], A[j, omega])   # residual sin el átomo j
U, s, Vt = np.linalg.svd(E, full_matrices=False)
D[:, j]      = U[:, 0]
A[j, omega]  = s[0] * Vt[0, :]
```

### Dónde entra la SVD
La SVD (`numpy.linalg.svd`, **no reimplementada**) se aplica al residual **restringido** a las señales
que usan el átomo (`omega`). Por el teorema de Eckart–Young, `s[0] u_1 v_1ᵀ` es la mejor aproximación
de rango 1 de `E` en norma de Frobenius; como la pareja anterior `(d_j, a_j)` también es una
aproximación de rango 1 de `E`, el error total **no aumenta** en cada actualización de átomo, y al
restringir a `omega` el soporte de `A` se conserva (no se pierde la dispersión). Restringir es
esencial: la SVD del residual completo llenaría la fila `j` de `A` de valores no nulos.

Pruebas específicas (`tests/test_modelo.py`):
* `test_rank1_svd_is_best_rank1_approximation`: `‖E − d aᵀ‖² = Σ_{k≥2} s_k²` y ninguna de 200
  alternativas de rango 1 (con coeficientes óptimos) es mejor.
* `test_ksvd_update_matches_literal_formula_for_first_atom`: el código coincide con la fórmula literal.
* `test_ksvd_update_restricted_support_and_monotone`: el soporte no crece y el error no aumenta.

## Inicialización — `init_dictionary(X, K, seed=43)`
Se eligen K columnas **no constantes** (norma > 1e-8) de `X_train`, sin reemplazo, con
`np.random.default_rng(43)`, y se normalizan. Se guarda como `D_init` en `modelo.npz` (y es
reproducible: la misma función con la misma semilla da el mismo `D_init`).

## Reinicialización de átomos sin uso
Si `omega` está vacío, el átomo se sustituye por la columna de **mayor norma** del residual de
entrenamiento actual `X − D A`, normalizada (sin repetir columna dentro de la misma pasada). Si
todos los residuales son prácticamente cero (≤ 1e-8), se conserva el átomo. Probado en
`test_unused_atom_is_reinitialized_with_normalized_residual` y `test_unused_atom_kept_when_residual_is_zero`.
En la ejecución real **no hubo átomos sin uso** (columna `unused_atoms` = 0 en todas las iteraciones).

## Parada
Número fijo de iteraciones (`n_iter = 8`), sin criterio de tolerancia. Cada iteración = codificación
OMP + una pasada de actualización por todos los átomos + registro de `train_mse`, `elapsed_s`,
`unused_atoms`. Tras la última iteración se **recalcula `A_train` con OMP sobre el D final**.

No se promete descenso estricto del MSE entre iteraciones: OMP es un algoritmo aproximado y puede
devolver una codificación peor que la anterior. Lo garantizado es el no-aumento dentro de cada
actualización de átomo. En esta ejecución el MSE sí bajó en las 8 iteraciones
(5.60e-4 → 3.38e-4; con `D_init` era 9.55e-4; tras recodificar con el D final, 3.41e-4).

## Validaciones (`validate_model`)
D (64, K) con normas 1 (desviación máx. 7.8e-16), todo finito, `A_train` (K, N), a lo más 4
coeficientes con |a| > 1e-10 por columna, `D @ A_train` con la forma de `X_train`, error finito.

## Bibliotecas
`numpy` (álgebra, SVD), `scikit-learn` (`orthogonal_mp`), `pandas` (historial),
`threadpoolctl` (límite de hilos BLAS). No se usa `DictionaryLearning`.

## Tiempo real y contingencia
Se intentó primero la configuración deseada (K=128, 1500 parches, 8 iteraciones) con un presupuesto
de 3600 s: tras la primera iteración se proyecta el tiempo total y sólo si excede el presupuesto se
baja de nivel (500 parches/5 iteraciones; después K=64). **Tiempo medido: ≈ 1.5 s** con 4 hilos
BLAS (el valor exacto de cada corrida está en `config_efectiva.json → training_time_s`). **No se
aplicó ninguna contingencia** (`contingency_level = 0`).

Observación: sin limitar hilos, en una máquina de 24 núcleos el mismo entrenamiento tardó 9–12 s por
sobresuscripción de OpenBLAS con matrices pequeñas; ver `00_general/incidencias.md`.

## Productos
| Archivo | Contenido |
|---|---|
| `modelo.npz` | `D` (64,128), `D_init` (64,128), `A_train` (128,1500) |
| `historial.csv` | `iteration, train_mse, elapsed_s, unused_atoms` |
| `config_efectiva.json` | configuración realmente usada, intentos de contingencia, tiempos, validaciones, hilos BLAS |
| `figuras/` | diccionario inicial vs final, diccionario final, curva de entrenamiento, uso de átomos |

## Limitaciones
* 1500 parches de una sola imagen y 8 iteraciones: diccionario pequeño (el artículo de denoising usa
  64 × 256 y más iteraciones).
* `T` fijo; no se usa el criterio de error del artículo de denoising.
* Inicialización con parches de datos (no DCT redundante).
* El resultado depende de la semilla; no se estudió la variabilidad entre semillas.
