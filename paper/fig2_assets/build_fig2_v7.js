// Fig.2 v7 — v5 structure (voting = highlighted panel beside the map panel),
// with a redrawn cone schematic (3D cone w/ elliptical mouth, gradient spheres,
// Gaussian weight lobe, escape-ray -> p_none) and NO labels on arrows.
const pptxgen = require("pptxgenjs");
const sharp = require("sharp");
const fs = require("fs");

const coneSvg = `<svg xmlns="http://www.w3.org/2000/svg" width="900" height="390">
<defs>
<radialGradient id="gi" cx="0.35" cy="0.3" r="0.95">
<stop offset="0" stop-color="#E7F6F1"/><stop offset="0.55" stop-color="#7CC9B4"/><stop offset="1" stop-color="#2E9C7C"/>
</radialGradient>
<radialGradient id="gj" cx="0.35" cy="0.3" r="0.95">
<stop offset="0" stop-color="#F7F8F9"/><stop offset="0.6" stop-color="#CDD2D8"/><stop offset="1" stop-color="#9AA1A8"/>
</radialGradient>
</defs>
<path d="M770,62 L832,84 L832,352 L770,330 Z" fill="#ECEDEF" stroke="#C9CDD4" stroke-width="2"/>
<path d="M70,195 L700,75 A26,120 0 0 1 700,315 Z" fill="#B07514" fill-opacity="0.10"/>
<ellipse cx="700" cy="195" rx="26" ry="120" fill="none" stroke="#B07514" stroke-opacity="0.45" stroke-width="2" stroke-dasharray="7 6"/>
<line x1="70" y1="195" x2="700" y2="75" stroke="#B07514" stroke-opacity="0.55" stroke-width="2.5"/>
<line x1="70" y1="195" x2="700" y2="315" stroke="#B07514" stroke-opacity="0.55" stroke-width="2.5"/>
<line x1="70" y1="195" x2="700" y2="115" stroke="#26323A" stroke-width="2" opacity="0.3"/>
<line x1="70" y1="195" x2="800" y2="149" stroke="#26323A" stroke-width="2" opacity="0.55"/>
<line x1="70" y1="195" x2="658" y2="232" stroke="#26323A" stroke-width="2" opacity="0.55"/>
<line x1="70" y1="195" x2="700" y2="275" stroke="#26323A" stroke-width="2" opacity="0.3"/>
<line x1="70" y1="195" x2="608" y2="195" stroke="#26323A" stroke-width="3.5" stroke-dasharray="11 8"/>
<path d="M380,133 C380,170 426,174 426,195 C426,216 380,220 380,257 Z" fill="#B07514" fill-opacity="0.16" stroke="#26323A" stroke-width="2.2"/>
<path d="M240,195 A170,170 0 0 0 237,163" fill="none" stroke="#26323A" stroke-width="2"/>
<circle cx="70" cy="195" r="8" fill="#26323A"/>
<circle cx="640" cy="182" r="40" fill="url(#gi)" stroke="#0F6E56" stroke-width="2.5"/>
<circle cx="665" cy="262" r="33" fill="url(#gj)" stroke="#6B7280" stroke-width="2.5"/>
<circle cx="610" cy="195" r="6.5" fill="#C0392B" stroke="#FFFFFF" stroke-width="1.5"/>
<circle cx="655" cy="233" r="6.5" fill="#C0392B" stroke="#FFFFFF" stroke-width="1.5"/>
<circle cx="800" cy="149" r="6.5" fill="#C0392B" stroke="#FFFFFF" stroke-width="1.5"/>
<g font-family="Georgia, 'Times New Roman', serif" font-style="italic" font-size="44" fill="#26323A">
<text x="24" y="260">p&#772;<tspan dy="12" font-size="32">k</tspan></text>
<text x="196" y="140">2.5&#963;<tspan dy="12" font-size="32">cal</tspan></text>
<text x="356" y="300">a<tspan dy="12" font-size="32">j</tspan></text>
<text x="624" y="196" fill="#FFFFFF" font-size="40">o<tspan dy="10" font-size="30">i</tspan></text>
<text x="650" y="276" fill="#FFFFFF" font-size="36">o<tspan dy="10" font-size="27">j</tspan></text>
<text x="500" y="378">unmapped &#8594; p<tspan dy="12" font-size="32">none</tspan></text>
</g>
</svg>`;

const NEU_B = "D3D7DC", NEU_T = "3F4750", INK = "1A1A1A", NOTE = "8A939C";
const AMB_F = "FCF4E4", AMB_B = "E8D5AC", AMB_T = "A66E10", AMB_C = "F5E6C4";
const HL_F = "FAF0DA", HL_B = "D9B97E";
const GRN_F = "EEF5EA", GRN_B = "BFD9B8", GRN_T = "33702A", GRN_C = "DCEBD5";
const TEA_F = "EBF4F2", TEA_B = "AFD3CB", TEA_T = "176B60", TEA_C = "CFE5DF";
const BLU_F = "EBF1F9", BLU_B = "B7CCE8", BLU_T = "24549C", BLU_C = "D6E3F4";
const NEU_C = "ECEDEF";

async function main() {
  const png = await sharp(Buffer.from(coneSvg)).resize(1800).png().toBuffer();
  fs.writeFileSync("cone_schematic_v2.png", png);

  const p = new pptxgen();
  p.defineLayout({ name: "FIG", width: 12, height: 4.66 });
  p.layout = "FIG";
  const s = p.addSlide();
  s.background = { color: "FFFFFF" };

  const box = (x, y, w, h, fill, border, bw) =>
    s.addShape("roundRect", { x, y, w, h, rectRadius: 0.09, fill: { color: fill }, line: { color: border, width: bw || 1.25 } });
  const chip = (t, x, y, w, h, fill, o) => {
    s.addShape("roundRect", { x, y, w, h, rectRadius: 0.05, fill: { color: fill }, line: { type: "none" } });
    if (t) s.addText(t, Object.assign({ x, y, w, h, fontFace: "Arial", fontSize: 11.5, color: INK,
      align: "center", valign: "middle", margin: 0.02, isTextBox: true }, o || {}));
  };
  const txt = (t, x, y, w, h, o) =>
    s.addText(t, Object.assign({ x, y, w, h, fontFace: "Arial", fontSize: 12, color: INK, margin: 0, isTextBox: true, valign: "top" }, o));
  const arrow = (x1, y1, x2, y2) =>
    s.addShape("line", { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
      flipH: x2 < x1, flipV: y2 < y1, line: { color: "000000", width: 1.5, endArrowType: "stealth" } });
  const M = (t, sub) => ({ text: t, options: sub ? { fontFace: "Cambria", italic: true, subscript: true } : { fontFace: "Cambria", italic: true } });

  // ---------- top strip ----------
  box(0.18, 0.07, 11.61, 0.70, "FFFFFF", NEU_B);
  txt("Offline, once: mapping & naming", 0.34, 0.12, 5, 0.22, { fontSize: 13, bold: true, italic: true, color: NEU_T });
  chip("phone capture", 0.36, 0.38, 1.24, 0.26, NEU_C);
  chip("COLMAP + board alignment", 1.83, 0.38, 2.25, 0.26, NEU_C);
  chip("3DGS (board frame)", 4.31, 0.38, 1.55, 0.26, NEU_C);
  chip("SAM cross-view instances", 6.09, 0.38, 1.95, 0.26, NEU_C);
  chip("named registry: 259 instances · 15 named", 8.27, 0.38, 3.25, 0.26, NEU_C);
  arrow(1.63, 0.51, 1.80, 0.51);
  arrow(4.11, 0.51, 4.28, 0.51);
  arrow(5.89, 0.51, 6.06, 0.51);
  arrow(8.07, 0.51, 8.24, 0.51);

  // ---------- wearer ----------
  box(0.18, 0.93, 2.36, 2.67, AMB_F, AMB_B);
  txt("Wearer gaze →\nworld fixations", 0.32, 1.02, 2.08, 0.46, { fontSize: 12.5, bold: true, italic: true, color: AMB_T });
  chip("fisheye ArUco PnP\n(wall tags)", 0.32, 1.58, 2.08, 0.40, AMB_C, { fontSize: 11 });
  chip("undistort · bias ·\npose → world ray", 0.32, 2.10, 2.08, 0.40, AMB_C, { fontSize: 11 });
  chip([{ text: "cluster → fixations ", options: {} }, M("F"), M("k", 1)],
       0.32, 2.62, 2.08, 0.30, AMB_C, { fontSize: 11 });
  txt("stable while walking", 0.32, 3.04, 2.08, 0.2, { align: "center", fontSize: 11, italic: true, color: NOTE });

  // ---------- conical posterior voting (core) ----------
  box(2.66, 0.93, 3.42, 2.67, HL_F, HL_B, 1.75);
  txt("Conical posterior voting", 2.66, 1.01, 3.42, 0.28, { align: "center", fontSize: 15, bold: true, italic: true, color: AMB_T });
  s.addImage({ path: "cone_schematic_v2.png", x: 2.80, y: 1.34, w: 3.14, h: 1.36 });
  txt([M("a"), M("j", 1), { text: " = exp(−", options: {} }, M("θ"), M("j", 1), { text: "²/2", options: {} },
       M("σ"), M("cal", 1), { text: "²),   ", options: {} },
       M("V"), M("i", 1), { text: " = Σ ", options: {} }, M("a"), M("j", 1), M("α"), M("j", 1)],
      2.80, 2.78, 3.14, 0.22, { align: "center", fontSize: 12 });
  txt([M("q"), M("i", 1), { text: " ∝ ", options: {} }, M("π"), M("i", 1), M("V"), M("i", 1),
       { text: ",   ", options: {} }, M("p"), M("none", 1), { text: " = 1 − Σ", options: {} },
       M("V"), M("i", 1), { text: "/", options: {} }, M("Z")],
      2.80, 3.02, 3.14, 0.22, { align: "center", fontSize: 12 });
  chip([M("q"), M("i", 1), { text: " ≥ ", options: {} }, M("τ"), { text: " → ", options: {} }, M("ô"),
        { text: "      else reject ∅", options: {} }],
       2.94, 3.28, 2.86, 0.26, AMB_C, { fontSize: 11.5 });

  // ---------- shared map ----------
  box(6.20, 0.93, 3.18, 2.67, TEA_F, TEA_B);
  txt("Shared 3DGS instance map", 6.20, 1.01, 3.18, 0.26, { align: "center", fontSize: 13, bold: true, italic: true, color: TEA_T });
  txt("one metric frame for human & robot", 6.20, 1.26, 3.18, 0.2, { align: "center", fontSize: 11.5, italic: true, color: TEA_T });
  s.addImage({ path: "hub_map.jpg", x: 6.34, y: 1.52, w: 2.90, h: 0.87 });
  chip("local depth + opacity inside the cone", 6.34, 2.55, 2.90, 0.26, TEA_C, { fontSize: 11 });
  chip("persistent instance IDs shared with robot", 6.34, 2.91, 2.90, 0.26, TEA_C, { fontSize: 11 });

  // ---------- robot ----------
  box(9.50, 0.93, 2.29, 3.61, BLU_F, BLU_B);
  txt("Robot execution", 9.64, 1.01, 2.0, 0.24, { fontSize: 13, bold: true, italic: true, color: BLU_T });
  chip("LIO-SAM + A* nav\ntags → world frame", 9.64, 1.34, 2.01, 0.40, BLU_C, { fontSize: 11 });
  chip("move · pick · place", 9.64, 1.86, 2.01, 0.28, BLU_C, { fontSize: 11 });
  chip("last-meter: hint → grasp cam", 9.64, 2.26, 2.01, 0.28, BLU_C, { fontSize: 10.5 });
  s.addShape("roundRect", { x: 9.72, y: 2.70, w: 1.85, h: 1.04, rectRadius: 0.04,
    fill: { color: "FFFFFF" }, line: { color: "9AA3AC", width: 1, dashType: "dash" } });
  txt("lastmeter.jpg", 9.72, 3.12, 1.85, 0.2, { align: "center", fontSize: 11.5, italic: true, color: NOTE });

  // ---------- speech ----------
  box(0.18, 3.76, 9.16, 0.78, GRN_F, GRN_B);
  txt("Speech → command", 0.34, 3.81, 3, 0.22, { fontSize: 13, bold: true, italic: true, color: GRN_T });
  chip("streaming ASR", 0.36, 4.07, 1.30, 0.26, GRN_C);
  chip("LLM parse: action + slots", 1.90, 4.07, 2.05, 0.26, GRN_C);
  chip("deictic word ↔ fixation window", 4.19, 4.07, 2.35, 0.26, GRN_C, { fontSize: 11 });
  chip([{ text: "confirm → ", options: {} }, M("c"), { text: " = (", options: {} }, M("a"),
        { text: ", ", options: {} }, M("ô"), { text: ", ", options: {} }, M("d"), { text: ")", options: {} }],
       6.84, 4.07, 2.05, 0.26, GRN_C, { fontSize: 11.5 });
  arrow(1.70, 4.20, 1.86, 4.20);
  arrow(3.99, 4.20, 4.15, 4.20);
  arrow(6.58, 4.20, 6.80, 4.20);

  // ---------- cross-panel arrows (no labels) ----------
  arrow(7.80, 0.79, 7.80, 0.91);            // offline -> map
  arrow(2.56, 2.30, 2.64, 2.30);            // wearer -> voting
  arrow(6.18, 2.30, 6.10, 2.30);            // map surfaces -> voting
  arrow(9.40, 1.80, 9.48, 1.80);            // map -> robot
  arrow(4.37, 3.62, 4.37, 3.74);            // voting -> binding
  arrow(9.36, 4.20, 9.48, 4.20);            // command -> robot

  await p.writeFile({ fileName: "fig2_skeleton_v7.pptx" });
  console.log("written fig2_skeleton_v7.pptx");
}
main().catch(e => { console.error(e); process.exit(1); });
