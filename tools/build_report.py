"""Genera el reporte (LaTeX -> PDF, más versión Markdown) y la presentación (PDF de diapositivas
con matplotlib, más guion Markdown) a partir de los archivos de resultados. Ningún número del
reporte se escribe a mano: todos se leen de 01_datos/, 02_modelo/ y 03_evaluacion/.

Uso:  python tools/build_report.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

REP = ROOT / "04_final" / "reporte"
PRE = ROOT / "04_final" / "presentacion"
FIG = ROOT / "04_final" / "figuras"


def load():
    cfg = json.loads((ROOT / "01_datos" / "config.json").read_text(encoding="utf-8"))
    eff = json.loads((ROOT / "02_modelo" / "config_efectiva.json").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "03_evaluacion" / "evaluacion_meta.json").read_text(encoding="utf-8"))
    hist = pd.read_csv(ROOT / "02_modelo" / "historial.csv")
    res = pd.read_csv(ROOT / "03_evaluacion" / "resultados.csv")
    vers = (ROOT / "01_datos" / "versiones.txt").read_text(encoding="utf-8")
    return cfg, eff, meta, hist, res, vers


def facts(cfg, eff, meta, hist, res):
    base = res[res.condition == "ruidosa_sin_procesar"].iloc[0]
    clean = res[res.condition == "limpia"].sort_values("T0")
    noisy = res[res.condition == "ruidosa"].sort_values("T0")
    bn = noisy.loc[noisy.psnr_db.idxmax()]
    bc = clean.loc[clean.psnr_db.idxmax()]
    clean_monotone = bool(np.all(np.diff(clean.psnr_db.values) > 0))
    noisy_monotone = bool(np.all(np.diff(noisy.psnr_db.values) > 0))
    hist_monotone = bool(np.all(np.diff(hist.train_mse.values) <= 0))
    return {
        "base_psnr": base.psnr_db, "base_mse": base.mse,
        "clean": clean, "noisy": noisy, "best_noisy": bn, "best_clean": bc,
        "gain": bn.psnr_db - base.psnr_db,
        "clean_monotone": clean_monotone, "noisy_monotone": noisy_monotone,
        "hist_monotone": hist_monotone,
        "mse_init": eff["train_mse_init_dictionary"], "mse_final": eff["train_mse_final_recoded"],
        "mse_iter1": hist.train_mse.iloc[0], "mse_last": hist.train_mse.iloc[-1],
        "t_train": eff["training_time_s"],
        "unused_total": int(hist.unused_atoms.sum()),
        "coded": res[res.T0 > 0],
    }


def results_rows(res):
    rows = []
    for r in res.itertuples():
        if r.condition == "ruidosa_sin_procesar":
            rows.append(("ruidosa (sin procesar)", "--", f"{r.mse:.6f}", f"{r.psnr_db:.2f}", "--", "--"))
        else:
            rows.append((r.condition, str(int(r.T0)), f"{r.mse:.6f}", f"{r.psnr_db:.2f}",
                         f"{r.mean_active_coefs:.2f}", f"{r.reconstruction_time_s * 1000:.1f}"))
    return rows


def conclusions(F, cfg):
    bn, bc = F["best_noisy"], F["best_clean"]
    c = []
    if F["clean_monotone"]:
        c.append(f"Con la imagen limpia, el PSNR crece monótonamente con T0 (de "
                 f"{F['clean'].psnr_db.iloc[0]:.2f} dB con T0={int(F['clean'].T0.iloc[0])} a "
                 f"{F['clean'].psnr_db.iloc[-1]:.2f} dB con T0={int(F['clean'].T0.iloc[-1])}): más átomos "
                 "representan mejor la señal.")
    else:
        c.append(f"Con la imagen limpia el mejor PSNR se obtuvo con T0={int(bc.T0)} ({bc.psnr_db:.2f} dB).")
    if F["noisy_monotone"]:
        c.append("Con la imagen ruidosa el PSNR también crece con T0 en el rango probado.")
    else:
        c.append(f"Con la imagen ruidosa el PSNR no es monótono: el máximo está en T0={int(bn.T0)} "
                 f"({bn.psnr_db:.2f} dB). Esto es consistente con un compromiso: con pocos átomos se pierde "
                 "detalle y con muchos OMP también ajusta parte del ruido (interpretación, no medida directa).")
    c.append(f"Frente a la imagen ruidosa sin procesar ({F['base_psnr']:.2f} dB), la mejor reconstrucción "
             f"mejora {F['gain']:.2f} dB. Es una ganancia real pero modesta; no se compara con el estado del arte.")
    d = F["coded"]
    share = float((d.coding_time_s / d.reconstruction_time_s).mean())
    exact = bool((d.mean_active_coefs == d.T0).all())
    c.append(("La actividad dispersa media coincide exactamente con T0" if exact else
              "La actividad dispersa media no supera T0") +
             f"; el tiempo de reconstrucción crece con T0 y la codificación OMP ocupa en promedio el "
             f"{100 * share:.0f}% de ese tiempo.")
    return c


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def build_tex(cfg, eff, meta, hist, res, vers, F):
    rows = "\n".join(" & ".join(tex_escape(x) for x in r) + r" \\" for r in results_rows(res))
    hrows = "\n".join(f"{int(h.iteration)} & {h.train_mse:.6e} & {h.elapsed_s:.2f} & {int(h.unused_atoms)} \\\\"
                      for h in hist.itertuples())
    concl = "\n".join(rf"\item {tex_escape(x)}" for x in conclusions(F, cfg))
    cont = ("No fue necesaria ninguna contingencia: se usó la configuración deseada "
            f"(K={eff['n_atoms']}, {eff['n_train_patches']} parches, {eff['n_iter']} iteraciones)."
            if eff["contingency_level"] == 0 else
            f"Se aplicó la contingencia de nivel {eff['contingency_level']}: {eff['contingency_label']}.")
    vers_tex = tex_escape(vers).replace("\n", r"\\" + "\n")
    return rf"""\documentclass[11pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=2.3cm]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{url}}
\renewcommand{{\figurename}}{{Figura}}
\renewcommand{{\tablename}}{{Tabla}}
\renewcommand{{\refname}}{{Referencias}}
\renewcommand{{\abstractname}}{{Resumen}}
\renewcommand{{\contentsname}}{{Contenido}}
\title{{Reconstrucción y eliminación de ruido en imágenes con K-SVD:\\ una adaptación didáctica}}
\author{{Proyecto académico --- Matemáticas Computacionales, INAOE}}
\date{{}}
\begin{{document}}
\maketitle
\begin{{abstract}}
Se implementa un ciclo K-SVD propio (codificación dispersa con \texttt{{orthogonal\_mp}} de scikit-learn y
actualización de átomos con \texttt{{numpy.linalg.svd}}) para aprender un diccionario de {eff['n_atoms']} átomos
a partir de {eff['n_train_patches']} parches $8\times8$ de la imagen \emph{{camera}}. El diccionario se usa
para reconstruir la imagen \emph{{coins}} (no vista en entrenamiento), limpia y con ruido gaussiano
$\sigma=20/255$, con dispersión $T_0\in\{{2,4,8\}}$. La mejor reconstrucción de la imagen ruidosa alcanza
{F['best_noisy'].psnr_db:.2f}~dB ($T_0={int(F['best_noisy'].T0)}$) frente a {F['base_psnr']:.2f}~dB de la
entrada ruidosa. Es una adaptación pequeña y didáctica; no es una réplica de los artículos originales ni
pretende superar el estado del arte.
\end{{abstract}}

\section{{Introducción y fundamento}}
Un parche $x\in\mathbb{{R}}^{{64}}$ se modela como combinación lineal de pocas columnas (átomos) de un
diccionario $D\in\mathbb{{R}}^{{64\times K}}$: $x\approx D\alpha$ con $\|\alpha\|_0\le T_0$. K-SVD
\cite{{aharon2006}} aprende $D$ alternando (i) codificación dispersa de los ejemplos con el diccionario actual y
(ii) actualización de los átomos uno por uno, junto con los coeficientes que los usan. Elad y Aharon
\cite{{elad2006}} usan esta idea para eliminar ruido: parches $8\times8$ con solapamiento, diccionario de
$64\times256$ (inicializado con un DCT redundante), OMP con criterio de error ($C=1.15$), $J=10$ iteraciones
de K-SVD y un promedio final con la imagen ruidosa ponderado por $\lambda=30/\sigma$.

Este proyecto toma una versión \textbf{{reducida y didáctica}}: diccionario entrenado en una imagen limpia
distinta a la de prueba, número fijo de átomos $T_0$ (no criterio de error), sin el término de fidelidad
$\lambda$ y con una sola imagen de prueba y una sola realización de ruido.

\section{{Datos y parches}}
\begin{{itemize}}
\item Entrenamiento: \texttt{{skimage.data.camera()}} a $256\times256$; prueba: \texttt{{skimage.data.coins()}}
  a $128\times128$ (\texttt{{img\_as\_float64}}, \texttt{{resize}} con \texttt{{anti\_aliasing=True}},
  \texttt{{preserve\_range=True}}).
\item Parches $8\times8$, paso 4, recorrido por filas y luego columnas, aplanado en orden C, incluyendo el último
  inicio válido. Camera produce {cfg['dimensions']['n_patches_train_total']} parches y coins
  {cfg['dimensions']['n_patches_test_total']}.
\item Cada parche se centra (se resta su media, que se guarda); no se normaliza por su norma.
\item Se filtran parches con norma centrada $\le10^{{-8}}$ y se eligen {cfg['n_train_patches']} sin reemplazo
  (RNG 42).
\item Prueba de ida y vuelta (extraer y reensamblar): error máximo
  {cfg['checks']['roundtrip_max_error_train']:.1e} (camera) y {cfg['checks']['roundtrip_max_error_test']:.1e}
  (coins), por debajo de $10^{{-10}}$.
\end{{itemize}}

\section{{Método}}
\subsection{{Codificación dispersa}}
$A=\arg\min\|Z-DA\|_F$ s.a. $\|a_i\|_0\le T$ se aproxima con OMP de scikit-learn
(\texttt{{orthogonal\_mp(D, Z, n\_nonzero\_coefs=T)}}), sin intercepto; las señales con norma $\le10^{{-8}}$
reciben coeficientes cero. $T$ es un máximo: OMP puede detenerse antes si el residual se anula.

\subsection{{Actualización K-SVD}}
Para cada átomo $j$, con $D$ y $A$ ya actualizados por los átomos anteriores:
\begin{{align*}}
\omega_j &= \{{i : A_{{j,i}}\neq0\}}, \qquad
E_j = X_{{:,\omega_j}} - D A_{{:,\omega_j}} + d_j\, A_{{j,\omega_j}},\\
E_j &= U\Sigma V^\top \;(\texttt{{numpy.linalg.svd}}), \qquad d_j\leftarrow u_1,\quad A_{{j,\omega_j}}\leftarrow \sigma_1 v_1^\top .
\end{{align*}}
La SVD se aplica al residual \emph{{restringido}} a las señales que usan el átomo, lo que conserva el soporte
de $A$; por Eckart--Young, $\sigma_1u_1v_1^\top$ es la mejor aproximación de rango 1 de $E_j$, de modo que el
error total no aumenta en cada actualización de átomo (verificado en las pruebas unitarias). Un átomo sin uso
se reinicializa con la columna de mayor norma del residual $X-DA$, normalizada; si todos los residuales son
prácticamente nulos se conserva. Tras la última iteración se recalcula $A_{{\text{{train}}}}$ con el $D$ final.

\subsection{{Configuración}}
$K={eff['n_atoms']}$ (diccionario {'sobrecompleto' if eff['overcomplete'] else 'NO sobrecompleto'}),
{eff['n_train_patches']} parches, {eff['n_iter']} iteraciones, $T={eff['train_sparsity']}$, inicialización con
columnas no constantes de $X_{{\text{{train}}}}$ normalizadas (RNG 43). {tex_escape(cont)}

\section{{Entrenamiento}}
El entrenamiento completo tardó {F['t_train']:.2f}~s en CPU (límite de 4 hilos BLAS). El MSE de representación
pasó de {F['mse_init']:.3e} con $D_{{\text{{init}}}}$ a {F['mse_final']:.3e} con el diccionario final
(Tabla~\ref{{tab:hist}}, Figura~\ref{{fig:dic}}). Átomos sin uso detectados en total: {F['unused_total']}.

\begin{{table}}[h]\centering
\caption{{Historial de entrenamiento (MSE tras la actualización del diccionario de cada iteración).}}\label{{tab:hist}}
\begin{{tabular}}{{rrrr}}\hline
iteración & MSE entrenamiento & tiempo acumulado (s) & átomos sin uso\\\hline
{hrows}
\hline\end{{tabular}}
\end{{table}}

\begin{{figure}}[h]\centering
\includegraphics[width=0.95\linewidth]{{../figuras/diccionarios.png}}
\caption{{Diccionario inicial (parches de entrenamiento) y diccionario final K-SVD.}}\label{{fig:dic}}
\end{{figure}}

\section{{Evaluación}}
Se reconstruye coins limpia y ruidosa ($\sigma=20/255$, una realización con RNG 44, sin clipping; $\sigma$
empírica {meta['noise_empirical_std']:.5f}). Para cada parche se codifica con OMP ($\le T_0$ átomos), se
restaura la media y se reensambla por promedio de solapamientos, sin clipping. Las métricas se calculan siempre
contra la imagen limpia. Verificación previa de métricas: imagen contra sí misma MSE $=0$, PSNR $=\infty$;
diferencia constante $0.1$: MSE $={meta['metric_checks']['const_0.1_mse']:.4f}$, PSNR
$={meta['metric_checks']['const_0.1_psnr']:.2f}$~dB. El tiempo es la mediana de {meta['n_timing_repeats']}
ejecuciones (extracción + OMP + reensamblado).

\begin{{table}}[h]\centering
\caption{{Resultados sobre coins $128\times128$ ({int(res.n_patches.dropna().iloc[0])} parches, K={eff['n_atoms']}).}}
\begin{{tabular}}{{lrrrrr}}\hline
entrada & $T_0$ & MSE & PSNR (dB) & coef. activos & tiempo (ms)\\\hline
{rows}
\hline\end{{tabular}}
\end{{table}}

\begin{{figure}}[h]\centering
\includegraphics[width=\linewidth]{{../figuras/reconstrucciones.png}}
\caption{{Reconstrucciones de la imagen limpia (arriba) y ruidosa (abajo) para cada $T_0$.}}
\end{{figure}}

\begin{{figure}}[!ht]\centering
\includegraphics[width=0.55\linewidth]{{../figuras/psnr_vs_t0.png}}
\caption{{PSNR frente a $T_0$. Línea discontinua: imagen ruidosa sin procesar.}}
\end{{figure}}

\section{{Discusión y conclusiones}}
\begin{{itemize}}
{concl}
\end{{itemize}}

\paragraph{{Limitaciones.}} Una sola imagen de entrenamiento y una de prueba, una realización de ruido, $T_0$
fijo en vez del criterio de error de \cite{{elad2006}}, sin promedio ponderado con la imagen ruidosa,
diccionario aprendido sobre parches limpios de otra imagen, coins redimensionada sin conservar su relación de
aspecto, y tiempos medidos en una sola máquina. Los resultados no son comparables directamente con las tablas
de los artículos.

\paragraph{{Reproducibilidad.}} \texttt{{python run\_all.py}} regenera todos los productos; \texttt{{pytest}}
verifica contratos y reproduce el pipeline en un directorio temporal. Versiones:\\
{{\small\ttfamily {vers_tex}}}

\begin{{thebibliography}}{{9}}
\bibitem{{aharon2006}} M. Aharon, M. Elad, A. Bruckstein. K-SVD: An Algorithm for Designing Overcomplete
Dictionaries for Sparse Representation. \emph{{IEEE Trans. Signal Processing}}, 54(11), 2006.
\url{{https://doi.org/10.1109/TSP.2006.881199}}
\bibitem{{elad2006}} M. Elad, M. Aharon. Image Denoising Via Sparse and Redundant Representations Over Learned
Dictionaries. \emph{{IEEE Trans. Image Processing}}, 15(12), 2006. \url{{https://doi.org/10.1109/TIP.2006.881969}}
\bibitem{{sk}} scikit-learn: \texttt{{sklearn.linear\_model.orthogonal\_mp}}; NumPy: \texttt{{numpy.linalg.svd}};
scikit-image: \texttt{{skimage.data}}, \texttt{{skimage.metrics}}.
\end{{thebibliography}}
\end{{document}}
"""


def build_md(cfg, eff, meta, hist, res, F):
    table = "| entrada | T0 | MSE | PSNR (dB) | coef. activos | tiempo (ms) |\n|---|---:|---:|---:|---:|---:|\n"
    table += "\n".join("| " + " | ".join(r) + " |" for r in results_rows(res))
    concl = "\n".join(f"- {x}" for x in conclusions(F, cfg))
    return f"""# Reconstrucción y eliminación de ruido con K-SVD — reporte

> Versión Markdown generada automáticamente por `tools/build_report.py` a partir de los archivos de resultados.
> La versión con formato es `reporte.pdf`.

## Configuración efectiva
K = {eff['n_atoms']} ({'sobrecompleto' if eff['overcomplete'] else 'NO sobrecompleto'}), {eff['n_train_patches']} parches,
{eff['n_iter']} iteraciones, T = {eff['train_sparsity']}, nivel de contingencia {eff['contingency_level']} ({eff['contingency_label']}).
Tiempo de entrenamiento: {F['t_train']:.2f} s. MSE de entrenamiento: {F['mse_init']:.3e} (D_init) → {F['mse_final']:.3e} (D final).

## Resultados (coins 128×128, σ = 20/255, una realización, RNG 44)
{table}

## Conclusiones
{concl}

![resumen](../figuras/resumen_resultados.png)
"""


# ---------------------------------------------------------------------------
# Presentación (PDF 16:9 con matplotlib)
# ---------------------------------------------------------------------------
def _slide(pdf, title, bullets=(), image=None, table=None):
    fig = plt.figure(figsize=(13.33, 7.5))
    fig.patch.set_facecolor("white")
    fig.text(0.05, 0.92, title, fontsize=26, weight="bold", color="#0b0b0b", va="top")
    fig.add_artist(plt.Line2D([0.05, 0.95], [0.85, 0.85], color="#2a78d6", lw=2))
    x_text = 0.05
    w_text = 0.9 if image is None and table is None else 0.38
    y = 0.80
    import textwrap
    for b in bullets:
        lines = textwrap.wrap(b, width=int(110 * w_text))
        fig.text(x_text, y, "•  " + "\n    ".join(lines), fontsize=15, color="#0b0b0b", va="top")
        y -= 0.065 * len(lines) + 0.03
    if image is not None:
        img = plt.imread(image)
        left = 0.46 if bullets else 0.05
        ax = fig.add_axes([left, 0.04, 0.95 - left, 0.78])
        ax.imshow(img); ax.axis("off")
    if table is not None:
        cols, rows = table
        ax = fig.add_axes([0.05 if not bullets else 0.46, 0.08, 0.9 if not bullets else 0.49, 0.7])
        ax.axis("off")
        t = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center")
        t.auto_set_font_size(False); t.set_fontsize(13); t.scale(1, 2.1)
        for (r, _), cell in t.get_celld().items():
            cell.set_edgecolor("#e4e3df")
            if r == 0:
                cell.set_text_props(weight="bold")
    pdf.savefig(fig); plt.close(fig)


def build_slides(cfg, eff, meta, hist, res, F):
    PRE.mkdir(parents=True, exist_ok=True)
    rows = [list(r) for r in results_rows(res)]
    cols = ["entrada", "T0", "MSE", "PSNR (dB)", "coef. activos", "tiempo (ms)"]
    slides = [
        ("Reconstrucción y eliminación de ruido con K-SVD",
         ["Adaptación didáctica pequeña de K-SVD (Aharon, Elad y Bruckstein, 2006; Elad y Aharon, 2006).",
          "Diccionario aprendido sobre parches 8×8 de camera; evaluación en coins (no vista).",
          "Proyecto de cuatro bloques: datos · modelo · evaluación · integración.",
          "Matemáticas Computacionales — INAOE"], None, None),
        ("Bloque 1 — Datos y parches",
         [f"camera → 256×256 (entrenamiento), coins → 128×128 (prueba).",
          f"Parches 8×8, paso 4, centrados: {cfg['dimensions']['n_patches_train_total']} (camera), "
          f"{cfg['dimensions']['n_patches_test_total']} (coins).",
          f"{cfg['n_train_patches']} parches de entrenamiento sin reemplazo (RNG 42).",
          f"Ida y vuelta extraer/reensamblar: error máx {cfg['checks']['roundtrip_max_error_train']:.1e}."],
         ROOT / "01_datos" / "figuras" / "muestra_parches.png", None),
        ("Bloque 2 — K-SVD propio",
         ["OMP: sklearn.linear_model.orthogonal_mp (≤ T átomos).",
          "Átomo j: SVD del residual restringido E_j = X_ω − D A_ω + d_j a_j; d_j ← u₁, a_j ← σ₁v₁ᵀ.",
          "Átomos sin uso → residual de mayor norma, normalizado.",
          f"K={eff['n_atoms']}, {eff['n_train_patches']} parches, {eff['n_iter']} iteraciones, T={eff['train_sparsity']}; "
          f"{F['t_train']:.2f} s en CPU; sin contingencia." if eff["contingency_level"] == 0 else
          f"Contingencia nivel {eff['contingency_level']}: {eff['contingency_label']}"],
         FIG / "diccionarios.png", None),
        ("Bloque 2 — Convergencia",
         [f"MSE de entrenamiento: {F['mse_init']:.2e} (D_init) → {F['mse_final']:.2e} (D final).",
          "No se garantiza descenso estricto entre iteraciones (OMP es aproximado); sí dentro de cada actualización de átomo."],
         FIG / "curva_entrenamiento.png", None),
        ("Bloque 3 — Resultados", [], None, (cols, rows)),
        ("Bloque 3 — Reconstrucciones", [], FIG / "reconstrucciones.png", None),
        ("Efecto de la dispersión T0", conclusions(F, cfg)[:2], FIG / "psnr_vs_t0.png", None),
        ("Conclusiones y limitaciones",
         conclusions(F, cfg)[2:] +
         ["Limitaciones: una imagen de prueba, una realización de ruido, T0 fijo (no criterio de error), sin "
          "promedio ponderado con la imagen ruidosa; no se afirma superar el estado del arte."], None, None),
    ]
    with PdfPages(PRE / "presentacion.pdf") as pdf:
        for title, bullets, image, table in slides:
            _slide(pdf, title, bullets, image, table)
    md = ["# Presentación — guion\n", "Generado por `tools/build_report.py`; diapositivas en `presentacion.pdf`.\n"]
    for k, (title, bullets, image, table) in enumerate(slides, 1):
        md.append(f"\n## {k}. {title}\n")
        md += [f"- {b}" for b in bullets]
        if image is not None:
            md.append(f"\n![]({os.path.relpath(image, PRE)})")
        if table is not None:
            md.append("\n| " + " | ".join(table[0]) + " |\n|" + "---|" * len(table[0]))
            md += ["| " + " | ".join(r) + " |" for r in table[1]]
    (PRE / "presentacion.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return len(slides)


def main():
    cfg, eff, meta, hist, res, vers = load()
    F = facts(cfg, eff, meta, hist, res)
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "reporte.tex").write_text(build_tex(cfg, eff, meta, hist, res, vers, F), encoding="utf-8")
    (REP / "reporte.md").write_text(build_md(cfg, eff, meta, hist, res, F), encoding="utf-8")
    n = build_slides(cfg, eff, meta, hist, res, F)
    print(f"presentación: {n} diapositivas -> {(PRE / 'presentacion.pdf').relative_to(ROOT)}")
    if shutil.which("pdflatex"):
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "reporte.tex"],
                               cwd=REP, capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-3000:])
                raise SystemExit("pdflatex falló")
        for ext in (".aux", ".log", ".out", ".toc"):
            (REP / f"reporte{ext}").unlink(missing_ok=True)
        print("reporte:", (REP / "reporte.pdf").relative_to(ROOT))
    else:
        print("pdflatex no disponible: se conserva reporte.tex y reporte.md")


if __name__ == "__main__":
    main()
