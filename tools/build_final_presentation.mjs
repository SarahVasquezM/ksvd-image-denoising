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

function addSpeaker(slide, name, block, color = C.blue) {
  addText(slide, name + " · " + block, { left: 770, top: 674, width: 445, height: 24 },
    { fontSize: 14, bold: true, color, alignment: "right" });
}

// 1. Portada
{
  const s = presentation.slides.add();
  s.background.fill = C.bg;
  addText(s, "K-SVD para reconstrucción y eliminación de ruido", { left: 72, top: 82, width: 550, height: 196 },
    { fontSize: 48, bold: true, color: C.navy });
  addText(s, "Implementación y resultados experimentales", { left: 76, top: 294, width: 520, height: 52 },
    { fontSize: 26, color: C.blue });
  addText(s, "Sarah Vásquez · datos y parches\nAlan (Edgar) · OMP y K-SVD\nSergio · experimentos y métricas\nAreli · integración y conclusiones",
    { left: 76, top: 390, width: 500, height: 130 }, { fontSize: 18, color: C.text });
  addText(s, "Matemáticas Computacionales · INAOE", { left: 76, top: 548, width: 500, height: 42 },
    { fontSize: 20, color: C.muted });
  await addImage(s, "04_final/figuras/reconstrucciones.png", { left: 650, top: 74, width: 560, height: 390 },
    "Reconstrucciones de la imagen coins limpia y ruidosa");
  addText(s, "26.12 dB", { left: 715, top: 492, width: 220, height: 66 },
    { fontSize: 42, bold: true, color: C.orange });
  addText(s, "mejor caso ruidoso · T0=4", { left: 717, top: 552, width: 360, height: 40 },
    { fontSize: 21, color: C.text });
  addText(s, "+4.03 dB frente a la entrada", { left: 717, top: 595, width: 390, height: 36 },
    { fontSize: 20, bold: true, color: C.blue });
  addSpeaker(s, "Sarah", "Integrante 1", C.blue);
  notes(s, [
    "RESPONSABLE: Sarah · TIEMPO SUGERIDO: 1 min 10 s.",
    "Abrir con el problema: observamos una imagen contaminada por ruido y queremos recuperar estructura sin suavizar indiscriminadamente los detalles. El proyecto usa representaciones dispersas: cada parche se aproxima con pocos átomos de un diccionario aprendido.",
    "Presentar al equipo y la división de trabajo que aparece en la diapositiva. Aclarar que cada bloque entrega archivos con contratos definidos, por lo que el resultado final es reproducible.",
    "Anticipar el resultado principal sin interpretarlo todavía: el mejor caso ruidoso obtuvo 26.12 dB con T0 igual a 4, una mejora de 4.03 dB frente a la observación. Indicar que es una prueba de concepto y que no se afirma superar el estado del arte.",
    "Transición: explicar primero cómo se prepararon las imágenes y los parches."
  ].join("\n\n"));
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
  addSpeaker(s, "Sarah", "Integrante 1", C.blue);
  notes(s, [
    "RESPONSABLE: Sarah · TIEMPO SUGERIDO: 1 min 50 s.",
    "Explicar que camera se usa exclusivamente para aprender el diccionario y coins para evaluar; así se evita entrenar con la imagen limpia exacta que después se reconstruye. Ambas imágenes se convierten a float64 antes de redimensionar.",
    "Describir el contrato: parches de 8 por 8, paso 4, recorrido por filas y aplanado C. Cada parche se centra restando su media, pero no se divide entre su norma. Se muestrean 1500 parches sin reemplazo con semilla 42.",
    "El ensamblado devuelve la media y promedia las contribuciones de los parches solapados. No se aplica clipping. La prueba de ida y vuelta produce un error máximo de 1.11 por 10 a la menos 16, lo que confirma posiciones, orden y cobertura.",
    "Cerrar señalando que los arreglos de datos son los mismos del proyecto de Alan/Edgar; las modificaciones añadidas se limitan a validaciones de entrada y compatibilidad de contratos.",
    "Transición: Alan explicará cómo se aprende el diccionario con esos parches."
  ].join("\n\n"));
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
  addSpeaker(s, "Alan (Edgar)", "Integrante 2", C.orange);
  notes(s, [
    "RESPONSABLE: Alan (Edgar) · TIEMPO SUGERIDO: 2 min.",
    "Definir el modelo: cada parche centrado z se aproxima como D por alfa, con a lo más T coeficientes distintos de cero. Como D tiene 128 átomos de dimensión 64, el diccionario es sobrecompleto.",
    "La primera etapa usa OMP. En cada paso se elige un átomo correlacionado con el residual y se recalculan los coeficientes sobre el soporte activo. No implementamos OMP desde cero; usamos orthogonal_mp de scikit-learn.",
    "La segunda etapa sí implementa el ciclo K-SVD. Para el átomo j se forma el residual sin su contribución y se restringe a las señales que actualmente lo usan. La mejor aproximación de rango uno se obtiene con la SVD: u1 actualiza el átomo y sigma1 por v1 actualiza sus coeficientes.",
    "Subrayar que restringir el residual conserva la dispersión. Un átomo sin uso se reinicializaría con el residual de mayor norma.",
    "Transición: mostrar ahora la evolución cuantitativa del entrenamiento."
  ].join("\n\n"));
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
  addText(s, "K=128 · 1500 parches\n8 iteraciones · CPU", { left: 86, top: 548, width: 310, height: 72 },
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
  addSpeaker(s, "Alan (Edgar)", "Integrante 2", C.orange);
  notes(s, [
    "RESPONSABLE: Alan (Edgar) · TIEMPO SUGERIDO: 2 min.",
    "Interpretar la curva como el MSE de representación después de actualizar el diccionario en cada iteración. Desciende de 5.60 por 10 a la menos 4 en la primera iteración a 3.38 por 10 a la menos 4 en la octava.",
    "La comparación lateral usa el diccionario inicial antes del entrenamiento, con MSE 9.55 por 10 a la menos 4, y el diccionario final después de recodificar todos los parches, con 3.41 por 10 a la menos 4. La reducción relativa es cercana al 64 por ciento.",
    "La recodificación final es necesaria porque los coeficientes almacenados durante la última actualización no necesariamente corresponden a una pasada completa de OMP con el diccionario definitivo.",
    "Ningún átomo quedó sin uso y no se activó la configuración de contingencia. El tiempo medido en esta máquina fue 2.09 segundos, pero debe presentarse sólo como referencia local.",
    "Transición: Sergio describirá cómo se evaluó el diccionario congelado."
  ].join("\n\n"));
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
  addSpeaker(s, "Sergio", "Integrante 3", C.blue);
  notes(s, [
    "RESPONSABLE: Sergio · TIEMPO SUGERIDO: 1 min 25 s.",
    "Explicar el protocolo antes de mostrar cifras. Se toma coins limpia como referencia y se genera una sola observación con ruido gaussiano de desviación 20 sobre 255 y semilla 44. El ruido se añade después del redimensionamiento.",
    "No se recorta la observación ni las reconstrucciones. El límite de cero a uno se usa únicamente al dibujar. Esto evita mejorar artificialmente el MSE mediante saturación.",
    "Se evalúan presupuestos T0 de 2, 4 y 8, tanto sobre la entrada limpia como sobre la ruidosa. Para cada caso se miden MSE, PSNR, coeficientes activos y mediana del tiempo de cinco repeticiones. La referencia ruidosa sin procesar completa los siete casos.",
    "Indicar visualmente que un soporte pequeño suaviza más y uno grande conserva más detalle, pero puede conservar también ruido.",
    "Transición: cuantificar esa observación con la tabla."
  ].join("\n\n"));
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
  addSpeaker(s, "Sergio", "Integrante 3", C.blue);
  notes(s, [
    "RESPONSABLE: Sergio · TIEMPO SUGERIDO: 1 min 35 s.",
    "Comenzar por la línea base: la entrada ruidosa tiene MSE 0.006179 y PSNR 22.09 dB. Todas las reconstrucciones dispersas mejoran esa referencia.",
    "En la imagen limpia, el PSNR aumenta de manera monotónica y alcanza 31.43 dB con ocho átomos. En la imagen ruidosa, el mejor valor es 26.12 dB con cuatro átomos; representa una ganancia de 4.03 dB y una reducción aproximada de 60.5 por ciento en MSE respecto de la entrada.",
    "Con ocho átomos el caso ruidoso baja a 25.64 dB. Interpretamos que la mayor capacidad reduce el error de aproximación limpio, pero también permite ajustar parte del ruido. Esta explicación es consistente con las mediciones, aunque no se midió directamente qué coeficientes corresponden al ruido.",
    "La tabla es editable en PowerPoint y proviene de resultados.csv.",
    "Transición: Areli conectará la tendencia con las conclusiones y limitaciones."
  ].join("\n\n"));
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
  addSpeaker(s, "Areli", "Integrante 4", C.orange);
  notes(s, [
    "RESPONSABLE: Areli · TIEMPO SUGERIDO: 1 min 35 s.",
    "Comparar las dos curvas. Para la entrada limpia, aumentar T0 amplía la capacidad de representación y el PSNR crece de 26.71 a 31.43 dB. Para la entrada ruidosa, la curva presenta un máximo en cuatro átomos y después pierde 0.48 dB.",
    "Interpretar el resultado como un compromiso entre sesgo y ajuste del ruido. Con dos átomos el modelo es restrictivo y puede eliminar estructura; con ocho explica más detalle, pero también componentes accidentales de la perturbación. El punto óptimo depende del nivel de ruido y no debe generalizarse a otras imágenes.",
    "La línea discontinua de 22.09 dB representa la observación sin procesar y permite ver que los tres presupuestos producen mejora en este experimento.",
    "Mencionar el costo: para la entrada ruidosa, el tiempo aumenta aproximadamente de 52 a 168 milisegundos entre T0 igual a 2 y 8; más costo no implica mejor restauración.",
    "Transición: cerrar con lo demostrado y con los límites del estudio."
  ].join("\n\n"));
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
  addText(s, "Siguiente evaluación", { left: 820, top: 145, width: 330, height: 46 },
    { fontSize: 29, bold: true, color: C.blue, alignment: "center" });
  addText(s, "más imágenes\n\nvarias semillas de ruido\n\nniveles adicionales de σ\n\ncriterio OMP por error\n\ncomparadores públicos",
    { left: 820, top: 205, width: 340, height: 300 }, { fontSize: 22, color: C.text, alignment: "center" });
  addText(s, "Conclusión: la dispersión debe ajustarse al nivel de ruido y validarse fuera de la prueba",
    { left: 82, top: 557, width: 1085, height: 56 }, { fontSize: 22, bold: true, color: C.orange, alignment: "center" });
  addSpeaker(s, "Areli", "Integrante 4", C.orange);
  notes(s, [
    "RESPONSABLE: Areli · TIEMPO SUGERIDO: 1 min 25 s.",
    "Cerrar con tres conclusiones. Primero, la implementación K-SVD funcionó y mejoró la observación ruidosa en 4.03 dB para este caso. Segundo, cuatro átomos equilibraron pérdida de detalle y ajuste del ruido; ocho fueron mejores sólo cuando la entrada estaba limpia. Tercero, el resultado es una prueba de concepto, no una evaluación generalizable.",
    "Explicar las limitaciones: una imagen de entrenamiento, una de prueba, una sola realización de ruido, redimensionamiento de coins y ausencia de comparadores externos. El método implementado tampoco incluye el término de fidelidad con la imagen ruidosa usado en el artículo de Elad y Aharon.",
    "Proponer como continuación varias imágenes, múltiples semillas y niveles de sigma, intervalos de confianza, una regla OMP basada en error y comparaciones bajo el mismo protocolo.",
    "Frase final sugerida: el aprendizaje de diccionarios ofrece una regularización interpretable, pero la dispersión debe seleccionarse de acuerdo con el ruido y validarse en datos independientes.",
    "Abrir el espacio para preguntas."
  ].join("\n\n"));
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
const finalizedPath = path.join(TMP_DIR, "presentacion_final_output.pptx");
const receiptPath = path.join(stagingDir, "presentacion_final.validation.json");
await fs.rm(finalizedPath, { force: true });
await fs.rm(receiptPath, { force: true });
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: finalizedPath,
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
  receiptPath,
});

await fs.copyFile(finalizedPath, FINAL_PPTX);
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
process.exitCode = 0;
