#!/usr/bin/env python3
"""fig_theta_page.py -- docs/E1_DATA/{theta_bins,sigma_curve}.json -> fig_theta_success.html。

    python Eye_Tracker/tools/fig_theta_page.py [out.html]

成功率-θ 曲线页(没答与答错同算失败),三种视图:配置对比 / 锥宽 σ 对比(对 θ)/ σ 曲线(对 σ,按 θ 档),
可剔除物理遮挡 trial;σ 竖线、乱猜线、分箱 n、悬停读数、数据表、明暗主题。源数据由
e4_table.rows_for 分箱(见会话脚本),这里只负责排版。
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/E1_DATA"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "fig_theta_success.html"

bins = json.load(open(DATA / "theta_bins.json", encoding="utf-8"))
curve = json.load(open(DATA / "sigma_curve.json", encoding="utf-8"))
MODES = {
    "config": {"label": "配置", "x": "theta",
               "series": [["full", "v1 冻结", "s1"], ["v2", "v2 capture 排序", "s2"],
                          ["v2selfcal3", "v2 会话连续自校准", "s3"], ["naive", "朴素最近质心", "s0"]]},
    "sigma": {"label": "锥宽 σ 对 θ", "x": "theta",
              "series": [["v2s10", "σ = 1.0°", "s1"], ["v2", "σ = 1.5°(v2)", "s2"],
                         ["v2s05", "σ = 0.5°", "s3"], ["v2s25", "σ = 2.5°", "s4"]]},
    "curve": {"label": "σ 曲线", "x": "sigma",
              "series": [["≥2.5°", "θ ≥ 2.5°", "s1"], ["1.0–2.5°", "θ 1.0–2.5°", "s2"],
                         ["<1.0°", "θ < 1.0°", "s3"], ["yield", "有目标判定的 final 占比", "s0"]]},
}
cfgs = sorted({c for m in ("config", "sigma") for c, _, _ in MODES[m]["series"]})
theta_data = {c: {v: [{k: b[k] for k in ("lo", "hi", "n", "hit", "rate", "ci")} for b in bins[c][v]]
                  for v in ("all", "no_occ")} for c in cfgs}
sig_x = [c["sigma"] for c in curve]
sig_data = {}
for name, _, _ in MODES["curve"]["series"]:
    if name == "yield":
        sig_data[name] = [{"rate": c["yield_finals"], "n": c["finals"], "hit": round(c["yield_finals"] * c["finals"]), "ci": None} for c in curve]
    else:
        sig_data[name] = [{"rate": c["tiers"][name]["rate"], "n": c["tiers"][name]["n"], "hit": c["tiers"][name]["hit"], "ci": c["tiers"][name]["ci"]} for c in curve]

NOTES = {
    "config": {"all": "两个最低分箱里 68 项有 44 项是 4 m 站位上目标被前排球挡住的 trial,任何方法都答不了,所以曲线在 1° 以下被压到乱猜附近。切到「剔除物理遮挡」看方法本身的极限。",
               "no_occ": "去掉物理遮挡后曲线基本单调:1.5° 以上接近全对,1° 到 1.5° 七到八成,1° 以下二十几项里 v2 与 v1 相当,会话连续自校准在 1 到 1.5° 多中 4 项、在 2.5 到 4° 少中 6 项(远卡偏置带进近卡的代价)。"},
    "sigma": {"all": "σ=1.0° 在每一档都不低于 1.5°,在 1 到 2.5° 档多中 3 项、1° 以下多中 4 项;0.5° 在 1° 以下与 1.0° 打平,但 2.5° 以上开始丢,射线打在球旁的桌面上;2.5° 全线偏低,邻居被收进锥里。",
              "no_occ": "剔除遮挡后同样:1.0° 与 1.5° 在 1.5° 以上都接近全对,分歧在 1.5° 以下,1.0° 略优;0.5° 从 2.5° 档就开始掉;2.5° 在 1 到 2.5° 档掉最多。"},
    "curve": {"all": "σ 有内部最优,在 1.0° 附近,即标定精度。往宽只是把邻居收进来;往窄先在易档丢分(射线落到球旁桌面),0.2° 时只有一半 final 还有目标判定,盯着没反馈。0.2° 到 0.5° 在 1° 以下档看似不差,是因为 2 到 3 秒的盯看协议只要有一段 final 打中就算命中,把召回的损失藏了起来。",
              "no_occ": ""},
}

html = r'''<meta charset="utf-8">
<title>角间隔成功率曲线</title>
<style>
.viz-root{color-scheme:light;
  --surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s0:#898781;--occ:rgba(11,11,11,.045)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
  --surface-1:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s0:#898781;--occ:rgba(255,255,255,.06)}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
  --surface-1:#1a1a19;--page:#0d0d0d;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s0:#898781;--occ:rgba(255,255,255,.06)}
html,body{margin:0}
body{background:var(--page,#f9f9f7)}
.viz-root{background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC",sans-serif;min-height:100vh;padding:28px 20px 48px;box-sizing:border-box}
.wrap{max-width:860px;margin:0 auto}
h1{font-size:20px;font-weight:600;margin:0 0 4px;letter-spacing:-.01em}
.sub{margin:0 0 18px;color:var(--ink-2);font-size:13.5px;max-width:64ch}
.filters{display:flex;gap:8px;align-items:center;margin:0 0 14px;flex-wrap:wrap}
.filters .lab{color:var(--muted);font-size:12px;letter-spacing:.06em;text-transform:uppercase;margin-right:4px}
.seg{display:inline-flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}
.seg button{background:transparent;border:0;color:var(--ink-2);font:inherit;font-size:13px;padding:6px 12px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--surface-1);color:var(--ink);font-weight:600;box-shadow:inset 0 0 0 1px var(--border)}
.seg button:focus-visible{outline:2px solid var(--s1);outline-offset:-2px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:16px 16px 10px;position:relative}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:0 0 6px;font-size:13px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.legend i{display:inline-block;width:18px;height:0;border-top:2px solid;border-radius:2px}
.legend b{font-weight:500;color:var(--ink-2)}
svg{display:block;width:100%;height:auto;overflow:visible}
.gl{stroke:var(--grid);stroke-width:1}
.ax{stroke:var(--axis);stroke-width:1}
.tk{fill:var(--muted);font-size:11.5px;font-variant-numeric:tabular-nums}
.tk.n{font-size:10.5px}
.lbl{fill:var(--ink-2);font-size:11.5px}
.ref{stroke:var(--axis);stroke-width:1}
.occ{fill:var(--occ)}
.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.mk{stroke:var(--surface-1);stroke-width:2}
.hit{fill:transparent;cursor:crosshair}
.cross{stroke:var(--axis);stroke-width:1;pointer-events:none}
.tip{position:absolute;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:12.5px;color:var(--ink);box-shadow:0 4px 14px rgba(0,0,0,.12);min-width:200px;display:none}
.tip .h{color:var(--ink-2);margin-bottom:4px;font-variant-numeric:tabular-nums}
.tip .r{display:flex;justify-content:space-between;gap:12px;font-variant-numeric:tabular-nums}
.tip .r i{display:inline-block;width:12px;height:0;border-top:2px solid;vertical-align:middle;margin-right:6px}
.note{color:var(--ink-2);font-size:12.5px;margin:10px 0 0;max-width:72ch}
details{margin-top:16px}
summary{cursor:pointer;color:var(--ink-2);font-size:13px}
table{border-collapse:collapse;margin-top:10px;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:5px 10px;text-align:right;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:500;font-size:11.5px;letter-spacing:.04em}
.tbl{overflow-x:auto}
</style>
<div class="viz-root"><div class="wrap">
<h1>成功率随角间隔与锥宽的变化</h1>
<p class="sub">每个卡项算一次:系统答对为成功,没答与答错都算失败。「配置」比 v1、v2 与自校准;「锥宽 σ 对 θ」比 v2 口径下四个 σ 的曲线;「σ 曲线」把横轴换成 σ,按 θ 档看最优锥宽。主集剔压力段与边走,有 θ 的 202 项。</p>
<div class="filters"><span class="lab">视图</span><div class="seg" role="group" aria-label="视图" id="modes"></div><span class="lab" id="trial-lab" style="margin-left:14px">trial</span><div class="seg" role="group" aria-label="是否剔除遮挡 trial" id="trialseg"><button id="b-all" aria-pressed="true">全部</button><button id="b-noocc" aria-pressed="false">剔除物理遮挡</button></div><span id="ninfo" style="color:var(--muted);font-size:12.5px"></span></div>
<div class="card">
<div class="legend" id="legend"></div>
<div id="chart"></div>
<div class="tip" id="tip"></div>
<p class="note" id="note"></p>
</div>
<details><summary>数据表</summary><div class="tbl" id="table"></div></details>
</div></div>
<script>
const THETA=__THETA__; const SIGX=__SIGX__; const SIG=__SIG__; const MODES=__MODES__; const NOTES=__NOTES__;
const BINS=THETA.full.all.map(b=>[b.lo,b.hi]);
const W=820,H=400,ML=52,MR=24,MT=18,MB=54;
let mode='config', view='all';
const css=v=>getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim();
const xlab=i=>{const [lo,hi]=BINS[i];return hi>=20?`≥${lo}°`:`${lo}–${hi}°`};
function points(){ // returns {X: labels, cells: series -> [ {rate,n,hit,ci} ]}
  const M=MODES[mode];
  if(M.x==='theta') return {labels:BINS.map((b,i)=>xlab(i)), sub:BINS.map((b,i)=>`n=${THETA.full[view][i].n}`), cells:Object.fromEntries(M.series.map(s=>[s[0],THETA[s[0]][view]]))};
  return {labels:SIGX.map(s=>`σ ${s}°`), sub:SIGX.map(()=>''), cells:Object.fromEntries(M.series.map(s=>[s[0],SIG[s[0]]]))};
}
function render(){
  const M=MODES[mode]; const P=points(); const K=P.labels.length;
  const px=i=>ML+(i+0.5)*(W-ML-MR)/K, py=v=>MT+(1-v)*(H-MT-MB);
  document.getElementById('trialseg').style.display=M.x==='theta'?'':'none';
  document.getElementById('trial-lab').style.display=M.x==='theta'?'':'none';
  const svg=[`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${M.label}">`];
  if(M.x==='theta'){const x0=ML,x1=ML+2*(W-ML-MR)/K;svg.push(`<rect class="occ" x="${x0}" y="${MT}" width="${x1-x0}" height="${H-MT-MB}"/>`);svg.push(`<text class="lbl" x="${x0+8}" y="${H-MB-10}">遮挡区 θ &lt; 0.96°</text>`);}
  for(const v of [0,.25,.5,.75,1]){svg.push(`<line class="gl" x1="${ML}" x2="${W-MR}" y1="${py(v)}" y2="${py(v)}"/>`);svg.push(`<text class="tk" x="${ML-8}" y="${py(v)+4}" text-anchor="end">${Math.round(v*100)}%</text>`);}
  svg.push(`<line class="ref" x1="${ML}" x2="${W-MR}" y1="${py(1/3)}" y2="${py(1/3)}"/>`);
  svg.push(`<text class="lbl" x="${W-MR-4}" y="${py(1/3)-5}" text-anchor="end">三球乱猜 33%</text>`);
  if(M.x==='theta'){const refs=mode==='sigma'?[[2,'σ 1.0°'],[3,'σ 1.5°']]:[[3,'σ = 1.5°(打分口径)']];
    for(const [k,lab] of refs){const xs=ML+k*(W-ML-MR)/K;svg.push(`<line class="ref" x1="${xs}" x2="${xs}" y1="${MT}" y2="${H-MB}"/>`);svg.push(`<text class="lbl" x="${xs+6}" y="${H-MB-8}">${lab}</text>`);}}
  else{const k=SIGX.indexOf(1.5);const xs=px(k);svg.push(`<line class="ref" x1="${xs}" x2="${xs}" y1="${MT}" y2="${H-MB}"/>`);svg.push(`<text class="lbl" x="${xs+6}" y="${MT+14}">打分口径 σ = 1.5°</text>`);}
  svg.push(`<line class="ax" x1="${ML}" x2="${W-MR}" y1="${py(0)}" y2="${py(0)}"/>`);
  P.labels.forEach((l,i)=>{svg.push(`<text class="tk" x="${px(i)}" y="${H-MB+18}" text-anchor="middle">${l}</text>`);if(P.sub[i])svg.push(`<text class="tk n" x="${px(i)}" y="${H-MB+33}" text-anchor="middle">${P.sub[i]}</text>`);});
  svg.push(`<text class="lbl" x="${(ML+W-MR)/2}" y="${H-4}" text-anchor="middle">${M.x==='theta'?'θ,目标到最近命名物的张角(度)':'σ,锥的角高斯宽度(度);半角 2σ'}</text>`);
  for(const [key,name,slot] of [...M.series].reverse()){const col=css('--'+slot);const pts=P.cells[key].map((b,i)=>b.rate==null?null:[px(i),py(b.rate)]).filter(Boolean);
    svg.push(`<path class="ln" stroke="${col}" d="${pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ')}"/>`);
    pts.forEach(p=>svg.push(`<circle class="mk" cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="4.5" fill="${col}"/>`));}
  svg.push(`<line class="cross" id="cross" x1="0" x2="0" y1="${MT}" y2="${H-MB}" style="display:none"/>`);
  P.labels.forEach((l,i)=>svg.push(`<rect class="hit" data-i="${i}" x="${ML+i*(W-ML-MR)/K}" y="${MT}" width="${(W-ML-MR)/K}" height="${H-MT-MB}" tabindex="0" aria-label="${l}"/>`));
  svg.push('</svg>'); document.getElementById('chart').innerHTML=svg.join('');
  const lg=document.getElementById('legend'); lg.textContent='';
  for(const [key,name,slot] of M.series){const s=document.createElement('span');const i=document.createElement('i');i.style.borderTopColor=css('--'+slot);const b=document.createElement('b');b.textContent=name;s.append(i,b);lg.append(s);}
  const tot=THETA.full[view].reduce((a,b)=>a+b.n,0);
  document.getElementById('ninfo').textContent=M.x!=='theta'?'主集 204 项,每个 σ 各回放 16 条录像':(view==='all'?`${tot} 项`:`${tot} 项(去掉 45 个被前球物理遮挡的 trial)`);
  document.getElementById('note').textContent=NOTES[mode][M.x==='theta'?view:'all'];
  const t=document.createElement('table');const thead=document.createElement('thead');const tr=document.createElement('tr');
  for(const h of [M.x==='theta'?'θ 分箱':'σ',...M.series.map(s=>s[1])]){const th=document.createElement('th');th.textContent=h;tr.append(th);}thead.append(tr);t.append(thead);
  const tb=document.createElement('tbody');
  P.labels.forEach((l,i)=>{const r=document.createElement('tr');const td0=document.createElement('td');td0.textContent=l;r.append(td0);
    for(const [key] of M.series){const d=P.cells[key][i];const td=document.createElement('td');td.textContent=d.n?`${d.hit}/${d.n} (${Math.round(d.rate*100)}%)`:'—';r.append(td);}tb.append(r);});
  t.append(tb);const tw=document.getElementById('table');tw.textContent='';tw.append(t);
  const tip=document.getElementById('tip'),cross=document.getElementById('cross'),card=document.querySelector('.card');
  const show=i=>{tip.textContent='';const h=document.createElement('div');h.className='h';h.textContent=P.labels[i]+(P.sub[i]?'  '+P.sub[i]:'');tip.append(h);
    for(const [key,name,slot] of M.series){const d=P.cells[key][i];const r=document.createElement('div');r.className='r';const l=document.createElement('span');const ic=document.createElement('i');ic.style.borderTopColor=css('--'+slot);l.append(ic,document.createTextNode(name));const v=document.createElement('span');v.textContent=d.n?`${d.hit}/${d.n} = ${Math.round(d.rate*100)}%`+(d.ci?`  [${Math.round(d.ci[0]*100)}–${Math.round(d.ci[1]*100)}]`:''):'—';r.append(l,v);tip.append(r);}
    cross.setAttribute('x1',px(i));cross.setAttribute('x2',px(i));cross.style.display='';
    const rect=card.getBoundingClientRect();const sr=document.querySelector('#chart svg').getBoundingClientRect();
    const cx=sr.left-rect.left+px(i)*sr.width/W;tip.style.display='block';const tw2=tip.offsetWidth;tip.style.left=Math.min(Math.max(8,cx-tw2/2),rect.width-tw2-8)+'px';tip.style.top=(sr.top-rect.top+MT*sr.height/H+8)+'px';};
  const hide=()=>{tip.style.display='none';cross.style.display='none';};
  document.querySelectorAll('.hit').forEach(el=>{const i=+el.dataset.i;el.addEventListener('pointermove',()=>show(i));el.addEventListener('pointerleave',hide);el.addEventListener('focus',()=>show(i));el.addEventListener('blur',hide);});
}
const modes=document.getElementById('modes');
for(const k of Object.keys(MODES)){const b=document.createElement('button');b.textContent=MODES[k].label;b.setAttribute('aria-pressed',k===mode?'true':'false');b.onclick=()=>{mode=k;[...modes.children].forEach(c=>c.setAttribute('aria-pressed',c===b?'true':'false'));render();};modes.append(b);}
document.getElementById('b-all').onclick=()=>{view='all';document.getElementById('b-all').setAttribute('aria-pressed','true');document.getElementById('b-noocc').setAttribute('aria-pressed','false');render();};
document.getElementById('b-noocc').onclick=()=>{view='no_occ';document.getElementById('b-noocc').setAttribute('aria-pressed','true');document.getElementById('b-all').setAttribute('aria-pressed','false');render();};
render();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',render);
new MutationObserver(render).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script>
'''
html = (html.replace("__THETA__", json.dumps(theta_data, ensure_ascii=False)).replace("__SIGX__", json.dumps(sig_x))
        .replace("__SIG__", json.dumps(sig_data, ensure_ascii=False)).replace("__MODES__", json.dumps(MODES, ensure_ascii=False))
        .replace("__NOTES__", json.dumps(NOTES, ensure_ascii=False)))
OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({len(html)} bytes)")
