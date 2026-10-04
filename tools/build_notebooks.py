"""Genera los cuatro notebooks (fuente) a partir de celdas definidas aquí.
Después se ejecutan con:  jupyter nbconvert --to notebook --execute --inplace <nb>
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

SETUP = r'''# Localiza la raíz del repositorio (local o Kaggle) y la añade a sys.path.
import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/SarahVasquezM/ksvd-image-denoising.git"
REPO_REF = os.environ.get("KSVD_REPO_REF", "main")

def find_root():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "src" / "modelo.py").exists():
            return p
    if Path("/kaggle/working").exists():          # Kaggle: clonar el repositorio
        dest = Path("/kaggle/working/ksvd-image-denoising")
        if not dest.exists():
            subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_REF,
                            REPO_URL, str(dest)], check=True)
        return dest
    raise FileNotFoundError("No se encontró la raíz del repositorio")

ROOT = find_root()
sys.path.insert(0, str(ROOT))
import warnings
# Mostrar advertencias sin rutas absolutas de la máquina (no se suprime ninguna).
warnings.formatwarning = lambda msg, cat, filename, lineno, line=None: (
    f"{cat.__name__} ({Path(filename).name}:{lineno}): {msg}\n")
from threadpoolctl import threadpool_limits
from src.pipeline import DEFAULT_THREADS, paths
threadpool_limits(limits=DEFAULT_THREADS)     # evita sobresuscripción BLAS con matrices pequeñas
P = paths(ROOT)
print("Raíz del repositorio:", ROOT.name)'''


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


def show(rel):
    return code(f'from IPython.display import Image, display\ndisplay(Image(filename=str(ROOT / "{rel}")))')


NB = {}

NB["01_datos/01_datos.ipynb"] = [
    md("# Bloque 1 — Datos y parches\n\nPrepara las imágenes (camera → entrenamiento 256×256, coins → prueba 128×128), "
       "extrae parches 8×8 con stride 4, los centra y muestrea 1500 parches de entrenamiento con RNG 42.\n\n"
       "Implementación: `src/preprocesamiento.py` (fuente única). Productos: `datos.npz`, `config.json`, "
       "`versiones.txt`, `figuras/`. Documentación: `nota_datos.md`."),
    code(SETUP),
    md("## 1. Generar los productos del bloque"),
    code("import json\nfrom src.pipeline import run_block1\nres = run_block1(ROOT)\nprint(json.dumps(res['summary'], indent=2))"),
    md("## 2. Prueba obligatoria de ida y vuelta (extraer → reensamblar)"),
    code('''import numpy as np
from src.preprocesamiento import extract_patches, assemble_patches, load_dataset
d = load_dataset(P["datos"])
for name in ("train_image", "test_image"):
    image = d[name]
    Z, means, positions = extract_patches(image)
    recovered = assemble_patches(Z + means[None, :], positions, image.shape)
    err = np.max(np.abs(recovered - image))
    print(f"{name}: Z{Z.shape}, means{means.shape}, positions{positions.shape}, error máx = {err:.3e}")
    assert err < 1e-10'''),
    md("## 3. Contrato de `datos.npz`"),
    code('''for k, v in d.items():
    print(f"{k:14s} shape={v.shape} dtype={v.dtype}")
Z, _, _ = extract_patches(d["train_image"])
assert np.array_equal(d["X_train"], Z[:, d["train_indices"]])
print("Índices referidos a la extracción completa:", Z.shape[1], "parches; únicos:", len(set(d["train_indices"])))'''),
    md("## 4. Configuración"),
    code('print(json.dumps({k: v for k, v in res["config"].items()}, indent=2, ensure_ascii=False))'),
    md("## 5. Figuras de inspección"),
    show("01_datos/figuras/imagenes_camera_coins.png"),
    show("01_datos/figuras/muestra_parches.png"),
]

NB["02_modelo/02_modelo.ipynb"] = [
    md("# Bloque 2 — OMP y K-SVD\n\nEntrena un diccionario de 128 átomos (64-D) con un ciclo K-SVD propio: "
       "codificación dispersa con `sklearn.linear_model.orthogonal_mp` y actualización átomo por átomo con "
       "`numpy.linalg.svd` sobre el residual restringido.\n\nImplementación: `src/modelo.py`. Productos: "
       "`modelo.npz`, `historial.csv`, `config_efectiva.json`, `figuras/`. Documentación: `nota_metodo.md`."),
    code(SETUP),
    md("## 1. Entrenamiento (mide primero; la contingencia sólo se activa si el tiempo proyectado excede el presupuesto)"),
    code('''import json, pandas as pd
from src.pipeline import run_block2
res = run_block2(ROOT)
eff = res["effective"]
print("Nivel de contingencia:", eff["contingency_level"], "-", eff["contingency_label"])
print(f"K={eff['n_atoms']}  N={eff['n_train_patches']}  iter={eff['n_iter']}  T={eff['train_sparsity']}  "
      f"tiempo={eff['training_time_s']:.2f}s")'''),
    code('pd.read_csv(P["historial"])'),
    md("## 2. Validaciones del modelo"),
    code('''print(json.dumps(eff["checks"], indent=2))
print("MSE con D_init:", eff["train_mse_init_dictionary"], " | MSE final recodificado:", eff["train_mse_final_recoded"])'''),
    md("## 3. Comprobación de la actualización de soporte fijo (SVD = mejor aproximación de rango 1)"),
    code('''import numpy as np
from src.modelo import load_model, sparse_code, ksvd_dictionary_update, rank1_atom_update
from src.preprocesamiento import load_dataset
X = load_dataset(P["datos"])["X_train"][:, :eff["n_train_patches"]]
m = load_model(P["modelo"])
D = m["D_init"]; A = sparse_code(D, X, eff["train_sparsity"])
j = int(np.argmax(np.sum(A != 0, axis=1)))
omega = np.flatnonzero(A[j] != 0)
E = X[:, omega] - D @ A[:, omega] + np.outer(D[:, j], A[j, omega])
d, a = rank1_atom_update(E)
s = np.linalg.svd(E, compute_uv=False)
print(f"átomo {j}: |omega|={omega.size}, ||E - d a^T||^2 = {np.linalg.norm(E - np.outer(d, a))**2:.6e}, "
      f"sum(s[1:]^2) = {np.sum(s[1:]**2):.6e}")
before = np.linalg.norm(X - D @ A)
D2, A2, info = ksvd_dictionary_update(X, D, A)
print(f"||X - DA||_F antes={before:.6f}  después de una pasada={np.linalg.norm(X - D2 @ A2):.6f}  info={info}")'''),
    md("## 4. Figuras"),
    show("02_modelo/figuras/diccionario_inicial_vs_final.png"),
    show("02_modelo/figuras/curva_entrenamiento.png"),
    show("02_modelo/figuras/uso_atomos.png"),
]

NB["03_evaluacion/03_experimentos.ipynb"] = [
    md("# Bloque 3 — Experimentos\n\nReconstruye `coins` (no usada en entrenamiento) limpia y con ruido gaussiano "
       "σ = 20/255 (una sola realización, RNG 44) para T0 ∈ {2, 4, 8}. Métricas contra la imagen limpia: MSE, "
       "PSNR (data_range = 1), coeficientes activos medios y tiempo.\n\nImplementación: `src/evaluacion.py`. "
       "Productos: `resultados.csv`, `reconstrucciones.npz`, `evaluacion_meta.json`, `figuras/`. "
       "Documentación: `nota_resultados.md`."),
    code(SETUP),
    md("## 1. Ejecutar experimentos (incluye las comprobaciones previas de métricas)"),
    code('''import json, pandas as pd
from src.pipeline import run_block3
df = run_block3(ROOT)
meta = json.loads((P["eval_dir"] / "evaluacion_meta.json").read_text(encoding="utf-8"))
print(json.dumps(meta["metric_checks"], indent=2))
print("σ empírica del ruido:", meta["noise_empirical_std"], " (nominal", 20/255, ")")
df'''),
    md("## 2. Lectura de resultados (calculada, no escrita a mano)"),
    code('''base = df[df.condition == "ruidosa_sin_procesar"].iloc[0]
for cond in ("limpia", "ruidosa"):
    sub = df[df.condition == cond].sort_values("T0")
    best = sub.loc[sub.psnr_db.idxmax()]
    print(f"{cond:8s}: PSNR por T0 =", dict(zip(sub.T0, sub.psnr_db.round(3))), f"-> mejor T0={int(best.T0)}")
noisy = df[df.condition == "ruidosa"]
print(f"Ruidosa sin procesar: {base.psnr_db:.3f} dB; ganancia máx: {noisy.psnr_db.max() - base.psnr_db:.3f} dB")'''),
    md("## 3. Control: diccionario inicial `D_init` frente al diccionario aprendido\n\n"
       "Análisis complementario (no forma parte de `resultados.csv`): misma imagen, mismo ruido y mismos T0, "
       "cambiando sólo el diccionario. Mide cuánto aporta el aprendizaje K-SVD frente a parches elegidos al azar."),
    code('''import numpy as np, warnings
from src.evaluacion import reconstruct_image, psnr, make_noisy
from src.modelo import load_model
from src.preprocesamiento import load_dataset, load_config
cfg = load_config(P["config"]); m = load_model(P["modelo"]); test = load_dataset(P["datos"])["test_image"]
noisy, _ = make_noisy(test, cfg["sigma"], cfg["seeds"]["noise"])
rows = []
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="OMP terminó", category=RuntimeWarning)  # sólo parada temprana
    for name in ("D_init", "D"):
        for T0 in cfg["eval_sparsities"]:
            rows.append({"diccionario": name, "T0": T0,
                         "psnr_limpia": psnr(test, reconstruct_image(test, m[name], T0)["image"]),
                         "psnr_ruidosa": psnr(test, reconstruct_image(noisy, m[name], T0)["image"])})
ctrl = pd.DataFrame(rows).pivot(index="T0", columns="diccionario")
ctrl["ganancia_limpia_dB"] = ctrl[("psnr_limpia", "D")] - ctrl[("psnr_limpia", "D_init")]
ctrl["ganancia_ruidosa_dB"] = ctrl[("psnr_ruidosa", "D")] - ctrl[("psnr_ruidosa", "D_init")]
ctrl.round(3)'''),
    md("## 4. Figuras"),
    show("03_evaluacion/figuras/reconstrucciones.png"),
    show("03_evaluacion/figuras/psnr_vs_t0.png"),
    show("03_evaluacion/figuras/mse_vs_t0.png"),
    show("03_evaluacion/figuras/actividad_tiempo.png"),
    show("03_evaluacion/figuras/mapas_error.png"),
]

NB["04_final/04_final.ipynb"] = [
    md("# Bloque 4 — Integración final\n\nLee los productos de los bloques 1–3 desde disco, verifica su coherencia, "
       "genera figuras finales, la tabla final, el reporte y la presentación."),
    code(SETUP),
    md("## 1. Verificación de integración"),
    code('''import json, pandas as pd
from src.pipeline import run_block4
out = run_block4(ROOT)
print(json.dumps(out["checks"], indent=2))'''),
    md("## 2. Tabla final"),
    code('out["table"]'),
    md("## 3. Configuración efectiva y contingencias"),
    code('''eff = json.loads(P["config_efectiva"].read_text(encoding="utf-8"))
{k: eff[k] for k in ("contingency_level", "contingency_label", "overcomplete", "n_atoms", "n_train_patches",
                     "n_iter", "training_time_s", "train_mse_init_dictionary", "train_mse_final_recoded")}'''),
    md("## 4. Figura-resumen"),
    show("04_final/figuras/resumen_resultados.png"),
    md("## 5. Reporte y presentación\n\nSe regeneran a partir de los archivos de resultados (`tools/build_report.py`). "
       "El PDF del reporte requiere `pdflatex`; si no está disponible se conserva el `.tex`/`.md`."),
    code('''import subprocess, sys
r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_report.py")], capture_output=True, text=True)
print(r.stdout[-2000:]); print(r.stderr[-2000:])'''),
]


def main():
    for rel, cells in NB.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, ROOT / rel)
        print("escrito", rel)


if __name__ == "__main__":
    main()
