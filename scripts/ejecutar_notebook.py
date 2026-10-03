"""Ejecuta en kernel nuevo y exporta el paquete local con integridad verificable."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import os
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
for variable, folder in [('IPYTHONDIR', 'ipython'), ('JUPYTER_CONFIG_DIR', 'jupyter'),
                         ('JUPYTER_RUNTIME_DIR', 'runtime'), ('MPLCONFIGDIR', 'matplotlib')]:
    path = ROOT / 'tmp' / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)
source = ROOT / 'notebooks' / '01_datos.ipynb'
notebook = nbformat.read(source, as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT)}})
# El kernel usa exactamente el intérprete del entorno activo, sin registro global.
client.create_kernel_manager()
client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client.execute()
output = ROOT / 'artifacts' / 'entrega_01'
nbformat.write(notebook, output / '01_datos.ipynb')
manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir() if p.is_file() and p.name != 'manifest.sha256.json'}
(output / 'manifest.sha256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
shutil.make_archive(str(ROOT / 'artifacts' / 'entrega_01_v1'), 'zip', output)
print('Notebook completo ejecutado en kernel nuevo:', output)
print((output / 'comprobaciones.json').read_text())
