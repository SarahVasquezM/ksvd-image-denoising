import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = process.env.WORKSPACE_DIR;
const SKILL_DIR = process.env.SKILL_DIR;
const TMP_DIR = process.env.TMP_DIR;
const FINAL_PPTX = process.env.FINAL_PPTX;
const RUNTIME_PYTHON = process.env.RUNTIME_PYTHON;
if (![workspaceDir, SKILL_DIR, TMP_DIR, FINAL_PPTX, RUNTIME_PYTHON].every(p => path.isAbsolute(p))) {
  throw new Error("All runtime paths must be absolute");
}

const {
  resolvePresentationFont,
  applyPresentationChartFont,
  finalizePresentation,
} = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
const family = resolvePresentationFont();
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

const C = {
  bg: "#F7F6F2",
  navy: "#17324D",
  blue: "#1F5A8A",
  pale: "#E7F0F6",
  orange: "#D76F3E",
  paleOrange: "#F7E5DC",
  text: "#17212B",
  muted: "#53616E",
  grid: "#D4DCE2",
  white: "#FFFFFF",
};

function addText(slide, text, pos, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: pos,
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: family,
    fontSize: style.fontSize ?? 24,
    bold: style.bold ?? false,
    color: style.color ?? C.text,
    alignment: style.alignment ?? "left",
    verticalAlignment: style.verticalAlignment ?? "middle",
    autoFit: "none",
  };
  return box;
}

function addTitle(slide, title, number) {
  addText(slide, title, { left: 68, top: 40, width: 1060, height: 66 },
    { fontSize: 36, bold: true, color: C.navy });
  addText(slide, String(number), { left: 1170, top: 44, width: 44, height: 34 },
    { fontSize: 17, bold: true, color: C.orange, alignment: "right" });
  addText(slide, "Matemáticas Computacionales · K-SVD", { left: 68, top: 678, width: 600, height: 22 },
    { fontSize: 14, color: C.muted });
}

async function addImage(slide, rel, pos, alt, fit = "contain") {
  const bytes = await fs.readFile(path.join(workspaceDir, rel));
  return slide.images.add({ blob: bytes, contentType: "image/png", alt, fit, position: pos });
}

function notes(slide, text) {
  slide.speakerNotes.textFrame.setText(text);
  slide.speakerNotes.setVisible(true);
}

// 1. Portada
{
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  addText(s, "K-SVD para reconstrucción y eliminación de ruido", { left: 72, top: 82, width: 550, height: 196 },
    { fontSize: 48, bold: true, color: C.navy });
  addText(s, "Implementación, resultados y auditoría final", { left: 76, top: 294, width: 500, height: 52 },
    { fontSize: 26, color: C.blue });
  addText(s, "Sarah Vásquez M.\nMatemáticas Computacionales · 3 de octubre de 2026", { left: 76, top: 520, width: 500, height: 84 },
    { fontSize: 20, color: C.muted });
  await addImage(s, "04_final/figuras/reconstrucciones.png", { left: 650, top: 74, width: 560, height: 390 },
    "Reconstrucciones de la imagen coins limpia y ruidosa");
  addText(s, "26.12 dB", { left: 715, top: 492, width: 220, height: 66 },
    { fontSize: 42, bold: true, color: C.orange });
  addText(s, "mejor caso ruidoso · T0=4", { left: 717, top: 552, width: 360, height: 40 },
    { fontSize: 21, color: C.text });
  addText(s, "+4.03 dB frente a la entrada", { left: 717, top: 595, width: 390, height: 36 },
    { fontSize: 20, bold: true, color: C.blue });
  notes(s, "Fuente: 03_evaluacion/resultados.csv y 04_final/figuras/reconstrucciones.png. Presentar el resultado como prueba de concepto, no como comparación con el estado del arte.");
}

// 2. Datos
{
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  addTitle(s, "Datos y contrato de parches", 2);
  addText(s, "• camera 256×256 para aprender\n\n• coins 128×128 para probar\n\n• parches 8×8 · paso 4 · orden C\n\n• 1500 muestras sin reemplazo · RNG 42\n\n• centrado por media · sin normalizar\n\n• ensamblado por promedio · sin clipping",
    { left: 76, top: 130, width: 470, height: 430 }, { fontSize: 23, color: C.text });
  await addImage(s, "01_datos/figuras/imagenes_camera_coins.png", { left: 570, top: 142, width: 620, height: 370 },
    "Imágenes camera y coins redimensionadas");
  addText(s, "Los cuatro arrays coinciden elemento a elemento con el paquete de Sarah.",
    { left: 575, top: 535, width: 600, height: 58 }, { fontSize: 24, bold: true, color: C.blue, alignment: "center" });
  addText(s, "Error máximo extraer → ensamblar: 1.11×10⁻¹⁶", { left: 575, top: 600, width: 600, height: 40 },
    { fontSize: 19, color: C.muted, alignment: "center" });
  notes(s, "Fuentes: 01_datos/config.json, 01_datos/datos.npz y comparación exacta con artifacts/entrega_01/datos.npz. El bloque 1 fue ejecutado en Kaggle CPU; los cuatro bloques se verificaron localmente.");
}

// 3. Método
{
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  addTitle(s, "Del parche al diccionario", 3);
  addText(s, "Imagen  →  parches centrados  →  OMP  →  actualización K-SVD  →  promedio de solapamientos",
    { left: 84, top: 124, width: 1110, height: 64 }, { fontSize: 24, bold: true, color: C.blue, alignment: "center" });
  addText(s, "OMP", { left: 88, top: 226, width: 190, height: 46 }, { fontSize: 28, bold: true, color: C.navy });
  addText(s, "a lo más T átomos\nsklearn.linear_model.orthogonal_mp",
    { left: 88, top: 275, width: 400, height: 90 }, { fontSize: 20, color: C.text });
  addText(s, "Actualización del átomo j", { left: 88, top: 385, width: 420, height: 68 },
    { fontSize: 27, bold: true, color: C.navy });
  addText(s, "Eⱼ = Xω − DAω + dⱼaⱼ\nEⱼ = UΣVᵀ\ndⱼ ← u₁   ·   aⱼ,ω ← σ₁v₁ᵀ",
    { left: 88, top: 442, width: 420, height: 132 }, { fontSize: 23, color: C.text });
  await addImage(s, "04_final/figuras/diccionarios.png", { left: 535, top: 224, width: 665, height: 280 },
    "Diccionario inicial y diccionario final K-SVD");
  addText(s, "K=128 átomos · dimensión 64 · 8 iteraciones · T=4 · CPU",
    { left: 545, top: 532, width: 640, height: 48 }, { fontSize: 23, bold: true, color: C.orange, alignment: "center" });
  addText(s, "La SVD se aplica al residual restringido de las señales que usan el átomo.",
    { left: 545, top: 590, width: 640, height: 44 }, { fontSize: 19, color: C.muted, alignment: "center" });
  notes(s, "Fuentes: src/modelo.py; Aharon, Elad y Bruckstein (2006), DOI 10.1109/TSP.2006.881199; NumPy SVD; scikit-learn orthogonal_mp.");
}

// 4. Entrenamiento — gráfico nativo
{
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  addTitle(s, "El entrenamiento redujo el error de representación", 4);
  addText(s, "9.55×10⁻⁴", { left: 82, top: 164, width: 300, height: 58 },
    { fontSize: 38, bold: true, color: C.navy });
  addText(s, "diccionario inicial", { left: 86, top: 222, width: 280, height: 32 },
    { fontSize: 19, color: C.muted });
  addText(s, "3.41×10⁻⁴", { left: 82, top: 312, width: 300, height: 58 },
    { fontSize: 38, bold: true, color: C.orange });
  addText(s, "diccionario final recodificado", { left: 86, top: 370, width: 330, height: 56 },
    { fontSize: 19, color: C.muted });
  addText(s, "64% menos MSE", { left: 82, top: 478, width: 330, height: 52 },
    { fontSize: 28, bold: true, color: C.blue });
  addText(s, "Sin contingencia\n32 pruebas aprobadas", { left: 86, top: 548, width: 310, height: 72 },
    { fontSize: 20, color: C.text });
  const chart = s.charts.add("line", {
    position: { left: 430, top: 142, width: 755, height: 470 },
    categories: ["1", "2", "3", "4", "5", "6", "7", "8"],
    series: [{
      name: "MSE de entrenamiento",
      values: [0.0005604473, 0.0004229563, 0.0003796942, 0.0003638848, 0.0003532115, 0.0003472041, 0.0003399084, 0.0003382060],
      line: { fill: C.blue, width: 3 },
      marker: { symbol: "circle", size: 8 },
    }],
    hasLegend: false,
    lineOptions: { smooth: false },
    xAxis: { title: "Iteración K-SVD", textStyle: { typeface: family, fontSize: 17, fill: C.muted }, line: { fill: C.grid, width: 1 } },
    yAxis: { title: "MSE", min: 0.00030, max: 0.00060, majorUnit: 0.00005, numberFormatCode: "0.00000", textStyle: { typeface: family, fontSize: 16, fill: C.muted }, majorGridlines: { fill: C.grid, width: 1 } },
    chartFill: C.bg,
    plotAreaFill: C.white,
    plotAreaLine: { fill: C.grid, width: 1 },
  });
  applyPresentationChartFont(chart, { fontFamily: family });
  notes(s, "Fuentes: 02_modelo/historial.csv y 02_modelo/config_efectiva.json. El porcentaje compara 9.549e-4 con 3.413e-4. El tiempo no se usa como resultado portátil.");
}

// 5. Evaluación
{
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  addTitle(s, "Protocolo de evaluación", 5);
  addText(s, "Una sola imagen de prueba · una sola realización de ruido · misma referencia limpia",
    { left: 90, top: 118, width: 1095, height: 48 }, { fontSize: 24, bold: true, color: C.blue, alignment: "center" });
  await addImage(s, "04_final/figuras/reconstrucciones.png", { left: 100, top: 178, width: 1080, height: 390 },
    "Cuadrícula de referencia, entrada ruidosa y seis reconstrucciones");
  addText(s, "σ=20/255 · RNG 44 · sin clipping", { left: 104, top: 588, width: 430, height: 38 },
    { fontSize: 21, bold: true, color: C.navy });
  addText(s, "T0 ∈ {2, 4, 8}", { left: 548, top: 588, width: 230, height: 38 },
    { fontSize: 21, bold: true, color: C.navy, alignment: "center" });
  addText(s, "MSE · PSNR · actividad · tiempo", { left: 805, top: 588, width: 380, height: 38 },
    { fontSize: 21, bold: true, color: C.navy, alignment: "right" });
  notes(s, "Fuentes: src/evaluacion.py, 03_evaluacion/evaluacion_meta.json y 04_final/figuras/reconstrucciones.png. Las métricas se calculan contra coins limpia con data_range=1.0.");
}

// 6. Tabla nativa
{
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  addTitle(s, "Resultados medidos", 6);
  const values = [
    ["Entrada", "T0", "MSE", "PSNR dB"],
    ["ruidosa sin procesar", "—", "0.006179", "22.09"],
    ["limpia", "2", "0.002132", "26.71"],
    ["limpia", "4", "0.001264", "28.98"],
    ["limpia", "8", "0.000719", "31.43"],
    ["ruidosa", "2", "0.002784", "25.55"],
    ["ruidosa", "4", "0.002442", "26.12"],
    ["ruidosa", "8", "0.002730", "25.64"],
  ];
  const table = s.tables.add({ rows: 8, columns: 4, left: 70, top: 145, width: 830, height: 430,
    columnWidths: [330, 120, 190, 190], values });
  table.borders.assign({ style: "solid", fill: C.grid, width: 1 });
  table.styleOptions = { headerRow: true, bandedRows: true };
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 4; c++) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? C.navy : (r === 6 ? C.paleOrange : (r % 2 === 0 ? C.pale : C.white));
      cell.text.style = { typeface: family, fontSize: r === 0 ? 19 : 18, bold: r === 0 || r === 6,
        color: r === 0 ? C.white : C.text, alignment: c === 0 ? "left" : "center" };
    }
  }
  addText(s, "T0=4", { left: 955, top: 196, width: 220, height: 64 },
    { fontSize: 42, bold: true, color: C.orange, alignment: "center" });
  addText(s, "mejor caso\nruidoso", { left: 955, top: 258, width: 220, height: 80 },
    { fontSize: 24, bold: true, color: C.navy, alignment: "center" });
  addText(s, "26.12 dB", { left: 955, top: 370, width: 220, height: 58 },
    { fontSize: 34, bold: true, color: C.blue, alignment: "center" });
  addText(s, "+4.03 dB\nsobre la entrada", { left: 955, top: 440, width: 220, height: 80 },
    { fontSize: 23, color: C.text, alignment: "center" });
  addText(s, "Tabla editable en PowerPoint", { left: 70, top: 603, width: 830, height: 32 },
    { fontSize: 16, color: C.muted, alignment: "center" });
  notes(s, "Fuente: 03_evaluacion/resultados.csv. La fila resaltada corresponde al mayor PSNR entre las reconstrucciones de la entrada ruidosa.");
}

// 7. PSNR — gráfico nativo
{
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  addTitle(s, "T0 controla el compromiso entre detalle y ruido", 7);
  addText(s, "Imagen limpia", { left: 78, top: 166, width: 300, height: 42 },
    { fontSize: 27, bold: true, color: C.blue });
  addText(s, "más átomos → mejor representación", { left: 78, top: 211, width: 330, height: 58 },
    { fontSize: 20, color: C.text });
  addText(s, "Imagen ruidosa", { left: 78, top: 320, width: 300, height: 42 },
    { fontSize: 27, bold: true, color: C.orange });
  addText(s, "máximo en T0=4\nT0=8 también ajusta ruido", { left: 78, top: 365, width: 330, height: 82 },
    { fontSize: 20, color: C.text });
  addText(s, "La explicación es interpretación; el máximo sí es una medición.", { left: 78, top: 515, width: 330, height: 78 },
    { fontSize: 18, color: C.muted });
  const chart = s.charts.add("line", {
    position: { left: 420, top: 144, width: 765, height: 475 },
    categories: ["2", "4", "8"],
    series: [
      { name: "entrada limpia", values: [26.71299, 28.982603, 31.430905], line: { fill: C.blue, width: 3 }, marker: { symbol: "circle", size: 9 } },
      { name: "entrada ruidosa", values: [25.553827, 26.121743, 25.638454], line: { fill: C.orange, width: 3 }, marker: { symbol: "circle", size: 9 } },
      { name: "ruidosa sin procesar", values: [22.09088, 22.09088, 22.09088], line: { fill: C.muted, width: 2, dash: "dash" }, marker: { symbol: "none" } },
    ],
    hasLegend: true,
    legend: { position: "top", overlay: false, textStyle: { typeface: family, fontSize: 16, fill: C.text } },
    lineOptions: { smooth: false },
    xAxis: { title: "T0 máximo por parche", textStyle: { typeface: family, fontSize: 17, fill: C.muted }, line: { fill: C.grid, width: 1 } },
    yAxis: { title: "PSNR dB", min: 21, max: 32, majorUnit: 2, numberFormatCode: "0", textStyle: { typeface: family, fontSize: 16, fill: C.muted }, majorGridlines: { fill: C.grid, width: 1 } },
    chartFill: C.bg,
    plotAreaFill: C.white,
    plotAreaLine: { fill: C.grid, width: 1 },
  });
  applyPresentationChartFont(chart, { fontFamily: family });
  notes(s, "Fuente: 03_evaluacion/resultados.csv. El máximo para la entrada ruidosa es T0=4 con 26.12 dB; la línea base es 22.09 dB.");
}

// 8. Cierre
{
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  addTitle(s, "Conclusiones y alcance de la evidencia", 8);
  addText(s, "1", { left: 76, top: 146, width: 44, height: 44 }, { fontSize: 30, bold: true, color: C.orange, alignment: "center" });
  addText(s, "K-SVD mejoró la entrada ruidosa en 4.03 dB para este caso.", { left: 140, top: 138, width: 600, height: 66 },
    { fontSize: 25, bold: true, color: C.navy });
  addText(s, "2", { left: 76, top: 252, width: 44, height: 44 }, { fontSize: 30, bold: true, color: C.orange, alignment: "center" });
  addText(s, "T0=4 equilibró detalle y ajuste de ruido; T0=8 fue mejor sólo en la entrada limpia.", { left: 140, top: 236, width: 600, height: 84 },
    { fontSize: 25, bold: true, color: C.navy });
  addText(s, "3", { left: 76, top: 366, width: 44, height: 44 }, { fontSize: 30, bold: true, color: C.orange, alignment: "center" });
  addText(s, "El resultado es una prueba de concepto: una imagen, una realización de ruido y sin comparación externa.", { left: 140, top: 348, width: 600, height: 94 },
    { fontSize: 25, bold: true, color: C.navy });
  addText(s, "Auditoría", { left: 820, top: 145, width: 330, height: 46 },
    { fontSize: 29, bold: true, color: C.blue, alignment: "center" });
  addText(s, "32 pruebas aprobadas\n\n4 notebooks ejecutados\n\narrays de Sarah idénticos\n\ncontrato de evaluación reparado\n\nbloque 1 Successful en Kaggle",
    { left: 820, top: 205, width: 340, height: 300 }, { fontSize: 22, color: C.text, alignment: "center" });
  addText(s, "Evidencia local completa · Kaggle verificado sólo para el bloque 1",
    { left: 82, top: 557, width: 1085, height: 56 }, { fontSize: 22, bold: true, color: C.orange, alignment: "center" });
  notes(s, "Fuentes: AUDITORIA.md, salida de pytest, notebooks ejecutados y docs/estado_plataformas.md. No atribuir una ejecución completa de los cuatro bloques a Kaggle.");
}

const requirements = {
  explicitTotalSlideCount: 8,
  requiredNativeTableOwnerSlides: [6],
  requiredNativeChartOwnerSlides: [4, 7],
  requiredEmbeddedWorkbookChartOwnerSlides: [],
  materializeLiteralChartWorkbooks: true,
};
const fontPolicy = { basis: "design", families: [family] };
const expectedSlideSizeEmu = "12192000,6858000";
const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "presentacion_final_candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", expectedSlideSizeEmu,
    "--validate-bullet-geometry",
    "--validate-heading-fit",
    "--require-native-table-slide", "6",
  ],
  requiredNativeTableOwnerSlides: [6],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "presentacion_final.validation.json"),
});

const finalDeck = await PresentationFile.importPptx(await FileBlob.load(FINAL_PPTX));
const previewDir = path.join(TMP_DIR, "previews");
await fs.mkdir(previewDir, { recursive: true });
for (let i = 0; i < finalDeck.slides.items.length; i++) {
  const slide = finalDeck.slides.items[i];
  const preview = await finalDeck.export({ slide, format: "png", scale: 1.25 });
  await fs.writeFile(path.join(previewDir, `slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(previewDir, `slide-${i + 1}.layout.json`), await layout.text());
}
const montage = await finalDeck.export({ format: "png", montage: true, scale: 0.55 });
await fs.writeFile(path.join(previewDir, "montage.png"), new Uint8Array(await montage.arrayBuffer()));
console.log(JSON.stringify({ family, final: FINAL_PPTX, result }, null, 2));
