# Colaboración y recepción del paquete

GitHub conserva código/documentación; Kaggle ejecuta notebooks y conserva versiones
con salidas. No hay sincronización automática configurada entre ambos servicios.
Cada integrante usa su notebook privado de trabajo; integrante 4 mantiene además
un notebook final. Evitar editar simultáneamente el mismo notebook.

## Permisos pendientes

| Integrante | GitHub | Kaggle | Permiso propuesto (pendiente de confirmar) |
|---|---|---|---|
| Dani (1) | SarahVasquezM | sarahvasquez97 | Propietaria de recursos |
| 2 | Pendiente | Pendiente | GitHub Write; lectura de paquete 1 en Kaggle |
| 3 | Pendiente | Pendiente | GitHub Write; lectura de paquetes 1 y 2 |
| 4 | Pendiente | Pendiente | GitHub Write; lectura de todos los paquetes |

Cada autor será propietario/editor de su notebook. Compartir acceso a notebook y
dataset por separado, verificando cada diálogo y destinatario. No asumir permisos
heredados ni que conocer una URL privada concede acceso. No se enviaron invitaciones.
Se necesitan usuarios exactos y confirmación de destinatarios/permisos por Dani.

Verificación de la interfaz actual: notebook tiene selector Privado/Público y
búsqueda de personas/grupos. Dataset tiene su propia sección Editar colaboradores
con búsqueda de usuarios/grupos. Ambos muestran a sarahvasquez97 como propietaria;
no se introdujeron destinatarios. Esto confirma controles separados, no invitaciones.

## Git

`main` es la versión funcional. Ramas sugeridas: `codex/dani-datos`,
`codex/integrante2-modelo`, `codex/integrante3-experimentos`,
`codex/integrante4-integracion`. Crear desde main actualizado, revisar diff,
probar el bloque y abrir PR pequeño; no modificar contratos unilateralmente.
No force push. No subir artifacts/, entornos, tokens, kaggle.json ni cachés.

## Kaggle

Crear un notebook Python privado, seleccionar acelerador None (CPU). Adjuntar el
recurso privado recibido. Importar el `.ipynb` de trabajo; el módulo `.py` permanece
en la entrada y es la única implementación. La primera celda comprueba imports y
busca el archivo real bajo `/kaggle/input/`. Si aparecen varias versiones, fija
SOURCE_DIR a una ruta de las que se imprimen; no adivines el nombre de montaje.
Ejecutar desde sesión reiniciada, guardar versión con salidas y anotar número/URL.

Cada autor exporta `/kaggle/working/entrega_0N/`. No sobrescribir una versión
aceptada; crear siguiente versión y pedir aceptación. El receptor adjunta la
versión identificada como entrada privada o utiliza salidas del notebook anterior
si su interfaz y permisos lo permiten. Integrante 4 registra versiones aceptadas.

Descargar el ZIP desde Output/Download después de guardar una versión con salidas,
guardar una copia local y verificar el manifiesto. El ZIP local de Dani también
sirve como respaldo independiente. Un guardado de fuente no demuestra ejecución.

## Carga para integrante 2

Tras adjuntar/descomprimir el paquete, localizar la ruta real:

```python
from pathlib import Path
import hashlib, json, sys
import numpy as np

matches = list(Path('/kaggle/input').rglob('datos.npz'))
print(matches)  # Si hay más de uno, elegir explícitamente la versión aceptada.
assert len(matches) == 1, 'Selecciona explícitamente el paquete acordado'
package = matches[0].parent
manifest = json.loads((package / 'manifest.sha256.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((package / name).read_bytes()).hexdigest() == expected, name
sys.path.insert(0, str(package))
from preprocesamiento import validate_data, extract_patches, assemble_patches
with np.load(package / 'datos.npz', allow_pickle=False) as f:
    arrays = dict(f)
config = json.loads((package / 'config.json').read_text())
validate_data(arrays)
X_train = arrays['X_train']
assert X_train.shape == (64, 1500)
```

En local sustituir sólo la búsqueda por `package = Path('ruta/entrega_01')`.
Si una versión de biblioteca produce diferencias de redondeo, conservar el paquete
recibido y sus versiones; investigar antes de regenerarlo o aceptar cambios.
No usar test_image para entrenar. La configuración efectiva del modelo la entrega
integrante 2; Dani no activa contingencias ni genera ese modelo.
