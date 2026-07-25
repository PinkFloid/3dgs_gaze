#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_seg_viewer.py -- 分割结果 -> 交互查看器(命名工作台)。

两种模式:
  点云模式(缺省,无 --splat):单文件 instance_viewer.html,实例按色相区分。
  splat 模式(--splat splat.ply):真高斯渲染(export_splat_from_ckpt.py 的输出),
    外观即 SuperSplat 级;labels 从 points.npz 按坐标精确匹配到每个高斯。
    数据落在伴随文件 instance_viewer_data.js(几十 MB,html 内嵌会撑爆)。

共同交互:左键旋转、滚轮缩放、右键平移;点击一团点/高斯选中实例 -> 侧栏
id/高斯数/视角数/尺寸 + thumbs 缩略图,输入名字回车即存(同名=合并);
"导出 names.json" 下载合并表,覆盖回 segmentation_sam/names.json 生效。

    python tools/export_seg_viewer.py                          # 点云版
    python tools/export_seg_viewer.py --splat <splat.ply>      # 真高斯版

浏览器双击 html 即可(thumbs 相对路径引用,与 html 同目录生效)。
"""
from __future__ import annotations

import argparse
import base64
import colorsys
import json
from pathlib import Path

import numpy as np

OBJ0 = 10  # label >= OBJ0 是物体;以下是地板/墙/天花板(与 lift_sam/grasp_intent 一致)
SH_C0 = 0.28209479177387814


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg-dir", default=str(root / "lab_result/segmentation_sam"))
    p.add_argument("--splat", default=None, help="splat.ply(给了就出真高斯渲染版)")
    p.add_argument("--max-splats", type=int, default=800000,
                   help="splat 模式高斯上限(带标签的优先保留,其余按不透明度取)")
    p.add_argument("--min-alpha", type=float, default=0.15, help="splat 模式不透明度下限")
    p.add_argument("--max-pts", type=int, default=600, help="每实例最多采样点数(拾取/点云)")
    p.add_argument("--bg-pts", type=int, default=30000, help="背景(地板墙)总采样点数")
    p.add_argument("--out", default=None, help="输出 html(默认 seg-dir/instance_viewer.html)")
    return p.parse_args()


def inst_color(i: int):
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.68, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def build_pick(xyz, label, inst_meta, rng, max_pts, bg_pts):
    """采样点 + 实例区间表(点云渲染与点击拾取共用)。"""
    parts, ranges = [], []
    bg_mask = label < OBJ0
    if bg_mask.sum() > 0:
        P = xyz[bg_mask]
        if len(P) > bg_pts:
            P = P[rng.choice(len(P), bg_pts, replace=False)]
        parts.append((P, np.tile(np.array([70, 70, 74], np.uint8), (len(P), 1))))
        ranges.append({"id": -1, "n": int(len(P))})
    for iid in sorted(inst_meta):
        m = label == iid
        n = int(m.sum())
        if n == 0:
            continue
        P = xyz[m]
        if len(P) > max_pts:
            P = P[rng.choice(len(P), max_pts, replace=False)]
        parts.append((P, np.tile(np.array(inst_color(iid), np.uint8), (len(P), 1))))
        it = inst_meta[iid]
        bmin, bmax = np.array(it["bbox_min"]), np.array(it["bbox_max"])
        ranges.append({
            "id": iid, "n": int(len(P)),
            "ng": int(it.get("n_gaussians", n)), "nv": int(it.get("n_views", 0)),
            "c": [round(float(v), 3) for v in it["centroid"]],
            "diag": round(float(np.linalg.norm(bmax - bmin)), 3),
        })
    P = np.concatenate([p for p, _ in parts])
    C = np.concatenate([c for _, c in parts])
    return P, C, ranges


def load_splat_ply(path):
    """export_splat_from_ckpt.py 的 ply -> (xyz, rgb01, alpha01, scale_m, quat)。"""
    f = open(path, "rb")
    header = []
    while True:
        line = f.readline().decode("ascii").strip()
        header.append(line)
        if line == "end_header":
            break
    props = [ln.split()[-1] for ln in header if ln.startswith("property")]
    n = int(next(ln for ln in header if ln.startswith("element vertex")).split()[-1])
    data = np.frombuffer(f.read(), dtype="<f4").reshape(n, len(props))
    ix = {p: i for i, p in enumerate(props)}
    xyz = data[:, [ix["x"], ix["y"], ix["z"]]].astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * data[:, [ix["f_dc_0"], ix["f_dc_1"], ix["f_dc_2"]]], 0, 1)
    alpha = 1.0 / (1.0 + np.exp(-data[:, ix["opacity"]]))
    scale = np.exp(data[:, [ix["scale_0"], ix["scale_1"], ix["scale_2"]]])
    quat = data[:, [ix["rot_0"], ix["rot_1"], ix["rot_2"], ix["rot_3"]]]
    quat = quat / np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-9)
    return xyz, rgb, alpha, scale, quat


def main():
    args = parse_args()
    seg = Path(args.seg_dir)
    out = Path(args.out) if args.out else seg / "instance_viewer.html"

    npz = np.load(seg / "points.npz")
    xyz, label = npz["xyz"].astype(np.float32), npz["label"].astype(np.int64)
    inst_meta = {int(i["id"]): i for i in json.load(open(seg / "instances.json", encoding="utf-8"))["instances"]}
    try:
        names = json.load(open(seg / "names.json", encoding="utf-8"))
    except Exception:
        names = {}
    if not isinstance(names, dict):
        names = {}
    # lift_sam 预生成的是全 id 空模板,空串=未命名,别当成已命名
    names = {k: v for k, v in names.items() if str(v).strip()}

    rng = np.random.default_rng(0)
    P, C, ranges = build_pick(xyz, label, inst_meta, rng, args.max_pts, args.bg_pts)
    lo, hi = P.min(0), P.max(0)
    scale = (hi - lo).max() / 65000.0
    Q = np.round((P - lo) / scale).astype(np.uint16)

    meta = {
        "lo": [float(v) for v in lo], "scale": float(scale),
        "ranges": ranges, "names": {str(k): v for k, v in names.items()},
        "n": int(len(P)),
    }

    if not args.splat:
        html = TEMPLATE.replace("__META__", json.dumps(meta, ensure_ascii=False)) \
                       .replace("__XYZ__", base64.b64encode(Q.tobytes()).decode()) \
                       .replace("__RGB__", base64.b64encode(C.tobytes()).decode())
        out.write_text(html, encoding="utf-8")
        print(f"viewer(点云): {out}  ({out.stat().st_size/1e6:.1f} MB, {len(P)} pts, {len(ranges)-1} instances)")
        return

    # ---------------- splat 模式 ----------------
    from scipy.spatial import cKDTree

    sx, srgb, salpha, sscale, squat = load_splat_ply(args.splat)
    print(f"splat.ply: {len(sx)} gaussians")

    d, i = cKDTree(xyz).query(sx, workers=-1)
    lbl = np.where(d < 1e-5, label[i], -1)          # 逐高斯标签(坐标精确匹配)
    ids = [r["id"] for r in ranges if r["id"] >= 0]
    idx_of = {iid: k + 1 for k, iid in enumerate(ids)}  # 0 = 背景/无标签
    lbl_idx = np.zeros(len(sx), np.uint16)
    for iid, k in idx_of.items():
        lbl_idx[lbl == iid] = k
    labeled = lbl_idx > 0
    print(f"matched to instances: {int(labeled.sum())} gaussians / {len(ids)} instances")

    keep = labeled | (salpha >= args.min_alpha)
    if keep.sum() > args.max_splats:
        budget = args.max_splats - int(labeled.sum())
        un = np.where(~labeled & keep)[0]
        un = un[np.argsort(-salpha[un])][:max(budget, 0)]
        keep = np.zeros(len(sx), bool)
        keep[un] = True
        keep |= labeled
    sx, srgb, salpha, sscale, squat, lbl_idx = (a[keep] for a in (sx, srgb, salpha, sscale, squat, lbl_idx))
    n = len(sx)
    print(f"kept: {n} gaussians")

    rec = np.zeros(n, dtype=[("p", "<f4", (3,)), ("s", "<f4", (3,)), ("q", "u1", (4,)),
                             ("c", "u1", (4,)), ("l", "<u2"), ("pad", "<u2")])
    rec["p"], rec["s"] = sx, sscale
    rec["q"] = np.clip(squat * 127 + 128, 0, 255).astype(np.uint8)
    rec["c"][:, :3] = np.clip(srgb * 255, 0, 255).astype(np.uint8)
    rec["c"][:, 3] = np.clip(salpha * 255, 0, 255).astype(np.uint8)
    rec["l"] = lbl_idx

    meta["splat"] = {"n": int(n), "maxIdx": len(ids)}
    data_js = ("window.__VMETA__=" + json.dumps(meta, ensure_ascii=False) + ";\n"
               + 'window.__PICK__="' + base64.b64encode(Q.tobytes()).decode() + '";\n'
               + 'window.__SPLAT__="' + base64.b64encode(rec.tobytes()).decode() + '";\n')
    data_path = out.parent / "instance_viewer_data.js"
    data_path.write_text(data_js, encoding="utf-8")
    out.write_text(SPLAT_TEMPLATE, encoding="utf-8")
    print(f"viewer(splat): {out} + {data_path.name} ({data_path.stat().st_size/1e6:.1f} MB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>instance viewer</title>
<style>
html,body{margin:0;height:100%;background:#101014;color:#e8e8ee;font:13px/1.5 "Microsoft YaHei",sans-serif;overflow:hidden}
#c{position:absolute;inset:0;width:100%;height:100%}
#side{position:absolute;top:0;right:0;width:300px;height:100%;background:#16161cee;padding:12px 14px;box-sizing:border-box;overflow-y:auto}
#side h2{font-size:14px;margin:0 0 8px}
.row{margin:6px 0}
input[type=text]{width:100%;box-sizing:border-box;background:#0c0c10;border:1px solid #333;color:#fff;padding:6px;border-radius:4px}
button{background:#2fa79b;border:none;color:#fff;padding:6px 10px;border-radius:4px;cursor:pointer;margin:2px 2px 2px 0}
button.gray{background:#3a3a44}
#thumbs img{width:100%;border-radius:4px;margin-top:6px}
#hint{position:absolute;left:10px;bottom:8px;color:#9a9aa8;font-size:12px}
label{user-select:none}
.kv{color:#9a9aa8}.kv b{color:#e8e8ee}
#named{max-height:180px;overflow-y:auto;font-size:12px;margin-top:4px;border-top:1px solid #2a2a33;padding-top:4px}
</style></head><body>
<canvas id="c"></canvas>
<div id="side">
  <h2>实例命名台 <span id="stats" class="kv"></span></h2>
  <div class="row"><label><input type="checkbox" id="bg"> 显示地板/墙</label>
      <label style="margin-left:10px"><input type="checkbox" id="dimNamed" checked> 淡化已命名</label></div>
  <div class="row kv">尺寸过滤 <span id="fLab"></span><br>
      <input type="range" id="fMin" min="0" max="60" value="0" style="width:46%">
      <input type="range" id="fMax" min="10" max="400" value="400" style="width:46%"></div>
  <div class="row"><input type="text" id="jump" placeholder="输 id 回车跳选 (如 141)"></div>
  <div id="sel" class="row kv">点击场景中的一团点选中实例</div>
  <div class="row"><input type="text" id="nameIn" placeholder="选中后输名字,回车保存"></div>
  <div class="row">
    <button id="next">下一个未命名</button>
    <button id="iso" class="gray">隔离(I)</button>
    <button id="exp">导出 names.json</button>
  </div>
  <div id="thumbs"></div>
  <div id="named"></div>
</div>
<div id="hint">左键旋转 · 滚轮缩放 · 右键平移 · 点击选实例 · I 隔离 · Esc 取消</div>
<script>
const META=__META__;
function dec(b){const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a.buffer}
const Q=new Uint16Array(dec("__XYZ__")),RGB=new Uint8Array(dec("__RGB__"));
const N=META.n,pos=new Float32Array(N*3);
for(let i=0;i<N*3;i+=3){pos[i]=Q[i]*META.scale+META.lo[0];pos[i+1]=Q[i+1]*META.scale+META.lo[1];pos[i+2]=Q[i+2]*META.scale+META.lo[2]}
const col=new Uint8Array(N*4);
for(let i=0;i<N;i++){col[i*4]=RGB[i*3];col[i*4+1]=RGB[i*3+1];col[i*4+2]=RGB[i*3+2];col[i*4+3]=255}
// ranges -> per-range start
let off=0;for(const r of META.ranges){r.s=off;off+=r.n}
const names=Object.assign({},META.names);
const byId={};for(const r of META.ranges)byId[r.id]=r;

const cv=document.getElementById("c"),gl=cv.getContext("webgl",{antialias:false});
const vs=`attribute vec3 p;attribute vec4 c;uniform mat4 mvp;uniform float ps;varying vec4 vc;
void main(){gl_Position=mvp*vec4(p,1.0);gl_PointSize=clamp(ps/gl_Position.w,1.5,9.0);vc=c;}`;
const fs=`precision mediump float;varying vec4 vc;void main(){if(vc.a<0.02)discard;gl_FragColor=vc;}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pr);gl.useProgram(pr);
const bp=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,bp);gl.bufferData(gl.ARRAY_BUFFER,pos,gl.STATIC_DRAW);
const ap=gl.getAttribLocation(pr,"p");gl.enableVertexAttribArray(ap);gl.vertexAttribPointer(ap,3,gl.FLOAT,false,0,0);
const bc=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,bc);gl.bufferData(gl.ARRAY_BUFFER,col,gl.DYNAMIC_DRAW);
const ac=gl.getAttribLocation(pr,"c");gl.enableVertexAttribArray(ac);gl.vertexAttribPointer(ac,4,gl.UNSIGNED_BYTE,true,0,0);
const umvp=gl.getUniformLocation(pr,"mvp"),ups=gl.getUniformLocation(pr,"ps");
gl.enable(gl.DEPTH_TEST);gl.clearColor(0.063,0.063,0.078,1);

const ctr=[(META.lo[0]+65000*META.scale/2),(META.lo[1]+65000*META.scale/2),1.0];
let tgt=[0,0,1],th=-1.1,ph=0.9,rad=9;
(function(){let sx=0,sy=0,sz=0,n=0;for(let i=0;i<N*3;i+=Math.max(3,(N/5000|0)*3)){sx+=pos[i];sy+=pos[i+1];sz+=pos[i+2];n++}tgt=[sx/n,sy/n,sz/n]})();
function mat(){
 const eye=[tgt[0]+rad*Math.cos(ph)*Math.cos(th),tgt[1]+rad*Math.cos(ph)*Math.sin(th),tgt[2]+rad*Math.sin(ph)];
 const f=norm3(sub3(tgt,eye)),r=norm3(cross(f,[0,0,1])),u=cross(r,f);
 const v=[r[0],u[0],-f[0],0,r[1],u[1],-f[1],0,r[2],u[2],-f[2],0,-dot(r,eye),-dot(u,eye),dot(f,eye),1];
 const a=cv.width/cv.height,fy=1.6,fx=fy/a,zn=0.05,zf=200;
 const p=[fx,0,0,0,0,fy,0,0,0,0,(zf+zn)/(zn-zf),-1,0,0,2*zf*zn/(zn-zf),0];
 return mul44(p,v)}
function sub3(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]]}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function norm3(a){const l=Math.hypot(a[0],a[1],a[2]);return[a[0]/l,a[1]/l,a[2]/l]}
function mul44(a,b){const o=new Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s}return o}
let M=null,dirty=true;
function draw(){
 if(cv.width!==cv.clientWidth*devicePixelRatio||cv.height!==cv.clientHeight*devicePixelRatio){cv.width=cv.clientWidth*devicePixelRatio;cv.height=cv.clientHeight*devicePixelRatio;gl.viewport(0,0,cv.width,cv.height)}
 M=mat();gl.uniformMatrix4fv(umvp,false,new Float32Array(M));gl.uniform1f(ups,cv.height*0.9);
 gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.drawArrays(gl.POINTS,0,N);dirty=false}
(function loop(){if(dirty)draw();requestAnimationFrame(loop)})();

let sel=null,iso=false;
function alphaPass(){
 const fmin=+document.getElementById("fMin").value/100,fmax=+document.getElementById("fMax").value/100;
 document.getElementById("fLab").textContent=`${(fmin*100)|0}cm – ${fmax>=4?"∞":((fmax*100)|0)+"cm"}`;
 const showBg=document.getElementById("bg").checked,dimNamed=document.getElementById("dimNamed").checked;
 for(const r of META.ranges){
  let a=255;
  if(r.id<0)a=showBg?90:0;
  else{
   if(r.diag<fmin||(fmax<4&&r.diag>fmax))a=0;
   else if(iso&&sel&&r.id!==sel.id)a=14;
   else if(sel&&r.id===sel.id)a=255;
   else if(dimNamed&&names[String(r.id)])a=48;
  }
  for(let i=r.s;i<r.s+r.n;i++)col[i*4+3]=a;
 }
 gl.bindBuffer(gl.ARRAY_BUFFER,bc);gl.bufferData(gl.ARRAY_BUFFER,col,gl.DYNAMIC_DRAW);dirty=true}
function updStats(){const t=META.ranges.filter(r=>r.id>=0).length,d=Object.keys(names).filter(k=>byId[+k]).length;
 document.getElementById("stats").textContent=`已命名 ${d}/${t}`}
function pick(mx,my){
 const w=cv.clientWidth,h=cv.clientHeight;let best=1e9,bi=-1;
 for(const r of META.ranges){if(r.id<0)continue;if(col[r.s*4+3]===0)continue;
  const step=Math.max(1,(r.n/220)|0);
  for(let i=r.s;i<r.s+r.n;i+=step){
   const x=pos[i*3],y=pos[i*3+1],z=pos[i*3+2];
   const cx=M[0]*x+M[4]*y+M[8]*z+M[12],cy=M[1]*x+M[5]*y+M[9]*z+M[13],cw=M[3]*x+M[7]*y+M[11]*z+M[15];
   if(cw<=0)continue;
   const sx=(cx/cw*0.5+0.5)*w,sy=(1-(cy/cw*0.5+0.5))*h;
   const d=(sx-mx)**2+(sy-my)**2;if(d<best){best=d;bi=r.id}}}
 return best<20*20?bi:-1}
function select(id){
 sel=id>=0?byId[id]:null;iso=false;alphaPass();
 const s=document.getElementById("sel"),tb=document.getElementById("thumbs");
 if(!sel){s.innerHTML="点击场景中的一团点选中实例";tb.innerHTML="";return}
 const nm=names[String(sel.id)]||"";
 s.innerHTML=`<b>#${sel.id}</b> ${nm?"= <b>"+nm+"</b>":"<i>(未命名)</i>"}<br>`+
  `高斯 <b>${sel.ng}</b> · 视角 <b>${sel.nv}</b> · 对角 <b>${(sel.diag*100).toFixed(0)}cm</b><br>`+
  `中心 (${sel.c[0]}, ${sel.c[1]}, ${sel.c[2]}) z=${sel.c[2]}`;
 document.getElementById("nameIn").value=nm;document.getElementById("nameIn").focus();
 tb.innerHTML=`<img src="thumbs/inst_${sel.id}_0.jpg" onerror="this.remove()">`+
              `<img src="thumbs/inst_${sel.id}_1.jpg" onerror="this.remove()">`}
function renderNamed(){
 const el=document.getElementById("named");
 el.innerHTML=Object.entries(names).sort((a,b)=>+a[0]-+b[0])
  .map(([k,v])=>`<div>#${k} = ${v}</div>`).join("");updStats()}

let drag=0,px=0,py=0;
cv.addEventListener("mousedown",e=>{drag=e.button===0?1:2;px=e.clientX;py=e.clientY});
addEventListener("mouseup",()=>drag=0);
addEventListener("mousemove",e=>{if(!drag)return;const dx=e.clientX-px,dy=e.clientY-py;px=e.clientX;py=e.clientY;
 if(drag===1){th-=dx*0.006;ph=Math.min(1.5,Math.max(-1.5,ph+dy*0.006))}
 else{const s=rad*0.0016,r=[Math.sin(th),-Math.cos(th),0];
  tgt[0]+=r[0]*-dx*s+Math.cos(th)*Math.cos(ph)*0;tgt[1]+=r[1]*-dx*s;tgt[2]+=dy*s}
 dirty=true});
cv.addEventListener("wheel",e=>{rad*=e.deltaY>0?1.12:0.89;rad=Math.min(60,Math.max(0.5,rad));dirty=true;e.preventDefault()},{passive:false});
cv.addEventListener("contextmenu",e=>e.preventDefault());
let downXY=null;
cv.addEventListener("pointerdown",e=>downXY=[e.clientX,e.clientY]);
cv.addEventListener("pointerup",e=>{if(!downXY)return;
 if(Math.hypot(e.clientX-downXY[0],e.clientY-downXY[1])<4&&e.button===0)select(pick(e.clientX,e.clientY));downXY=null});
addEventListener("keydown",e=>{
 if(e.target.tagName==="INPUT"&&e.key!=="Escape")return;
 if(e.key==="i"||e.key==="I"){iso=!iso;alphaPass()}
 if(e.key==="Escape"){select(-1)}});
document.getElementById("nameIn").addEventListener("keydown",e=>{
 if(e.key!=="Enter"||!sel)return;const v=e.target.value.trim();
 if(v)names[String(sel.id)]=v;else delete names[String(sel.id)];
 renderNamed();alphaPass();select(sel.id)});
document.getElementById("jump").addEventListener("keydown",e=>{
 if(e.key!=="Enter")return;const id=parseInt(e.target.value);if(byId[id])select(id)});
document.getElementById("next").onclick=()=>{
 const un=META.ranges.filter(r=>r.id>=0&&!names[String(r.id)]&&col[r.s*4+3]>0).sort((a,b)=>b.diag-a.diag);
 if(un.length)select(un[0].id)};
document.getElementById("iso").onclick=()=>{iso=!iso;alphaPass()};
document.getElementById("exp").onclick=()=>{
 const blob=new Blob([JSON.stringify(names,null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="names.json";a.click()};
for(const id of["bg","dimNamed","fMin","fMax"])document.getElementById(id).addEventListener("input",alphaPass);
alphaPass();renderNamed();
</script></body></html>
"""

SPLAT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>instance viewer (splat)</title>
<style>
html,body{margin:0;height:100%;background:#101014;color:#e8e8ee;font:13px/1.5 "Microsoft YaHei",sans-serif;overflow:hidden}
#c{position:absolute;inset:0;width:100%;height:100%}
#side{position:absolute;top:0;right:0;width:300px;height:100%;background:#16161cee;padding:12px 14px;box-sizing:border-box;overflow-y:auto}
#side h2{font-size:14px;margin:0 0 8px}
.row{margin:6px 0}
input[type=text]{width:100%;box-sizing:border-box;background:#0c0c10;border:1px solid #333;color:#fff;padding:6px;border-radius:4px}
button{background:#2fa79b;border:none;color:#fff;padding:6px 10px;border-radius:4px;cursor:pointer;margin:2px 2px 2px 0}
button.gray{background:#3a3a44}
#thumbs img{width:100%;border-radius:4px;margin-top:6px}
#hint{position:absolute;left:10px;bottom:8px;color:#9a9aa8;font-size:12px}
#load{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);color:#9a9aa8}
label{user-select:none}
.kv{color:#9a9aa8}.kv b{color:#e8e8ee}
#named{max-height:180px;overflow-y:auto;font-size:12px;margin-top:4px;border-top:1px solid #2a2a33;padding-top:4px}
</style></head><body>
<canvas id="c"></canvas><div id="load">加载高斯数据…</div>
<div id="side">
  <h2>实例命名台 <span id="stats" class="kv"></span></h2>
  <div class="row"><label><input type="checkbox" id="bg" checked> 显示地板/墙</label>
      <label style="margin-left:10px"><input type="checkbox" id="dimNamed"> 淡化已命名</label></div>
  <div class="row kv">尺寸过滤 <span id="fLab"></span><br>
      <input type="range" id="fMin" min="0" max="60" value="0" style="width:46%">
      <input type="range" id="fMax" min="10" max="400" value="400" style="width:46%"></div>
  <div class="row"><input type="text" id="jump" placeholder="输 id 回车跳选 (如 141)"></div>
  <div id="sel" class="row kv">点击一个物体选中实例</div>
  <div class="row"><input type="text" id="nameIn" placeholder="选中后输名字,回车保存"></div>
  <div class="row">
    <button id="next">下一个未命名</button>
    <button id="iso" class="gray">隔离(I)</button>
    <button id="exp">导出 names.json</button>
  </div>
  <div id="thumbs"></div>
  <div id="named"></div>
</div>
<div id="hint">左键旋转 · 滚轮缩放 · 右键平移 · 点击选实例 · I 隔离 · Esc 取消</div>
<script src="instance_viewer_data.js"></script>
<script>
const META=window.__VMETA__;
function dec(b){const s=atob(b),a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a.buffer}
// ---- 拾取点(u16 量化) ----
const Q=new Uint16Array(dec(window.__PICK__));
const NP=META.n,ppos=new Float32Array(NP*3);
for(let i=0;i<NP*3;i+=3){ppos[i]=Q[i]*META.scale+META.lo[0];ppos[i+1]=Q[i+1]*META.scale+META.lo[1];ppos[i+2]=Q[i+2]*META.scale+META.lo[2]}
let off=0;for(const r of META.ranges){r.s=off;off+=r.n}
const names=Object.assign({},META.names);
const byId={};const idToIdx={};let k1=0;
for(const r of META.ranges){byId[r.id]=r;if(r.id>=0){k1++;idToIdx[r.id]=k1;r.idx=k1}}
const KMAX=k1,vis=new Uint8Array(KMAX+1).fill(1);
// ---- splat 数据 -> worker ----
const SP=META.splat,NS=SP.n,TW=4096,TH=Math.ceil(NS*4/TW);
const sbuf=dec(window.__SPLAT__);
const wsrc=`
let N=0,cx=null,depth=null,qb=null;
function covOf(sx,sy,sz,qw,qx,qy,qz){
 const xx=qx*qx,yy=qy*qy,zz=qz*qz,xy=qx*qy,xz=qx*qz,yz=qy*qz,wx=qw*qx,wy=qw*qy,wz=qw*qz;
 const R=[1-2*(yy+zz),2*(xy-wz),2*(xz+wy), 2*(xy+wz),1-2*(xx+zz),2*(yz-wx), 2*(xz-wy),2*(yz+wx),1-2*(xx+yy)];
 const M=[R[0]*sx,R[1]*sy,R[2]*sz, R[3]*sx,R[4]*sy,R[5]*sz, R[6]*sx,R[7]*sy,R[8]*sz];
 return [M[0]*M[0]+M[1]*M[1]+M[2]*M[2], M[0]*M[3]+M[1]*M[4]+M[2]*M[5], M[0]*M[6]+M[1]*M[7]+M[2]*M[8],
         M[3]*M[3]+M[4]*M[4]+M[5]*M[5], M[3]*M[6]+M[4]*M[7]+M[5]*M[8], M[6]*M[6]+M[7]*M[7]+M[8]*M[8]]}
onmessage=e=>{
 const d=e.data;
 if(d.type=='init'){
  N=d.n;const f=new Float32Array(d.buf),u8=new Uint8Array(d.buf),u16=new Uint16Array(d.buf);
  cx=new Float32Array(N*3);depth=new Float32Array(N);qb=new Uint32Array(N);
  const tex=new Float32Array(d.tw*d.th*4);
  for(let i=0;i<N;i++){
   const b=i*9,by=i*36;
   const x=f[b],y=f[b+1],z=f[b+2];cx[i*3]=x;cx[i*3+1]=y;cx[i*3+2]=z;
   const qw=(u8[by+24]-128)/127,qx=(u8[by+25]-128)/127,qy=(u8[by+26]-128)/127,qz=(u8[by+27]-128)/127;
   const C=covOf(f[b+3],f[b+4],f[b+5],qw,qx,qy,qz);
   const t=i*16;
   tex[t]=x;tex[t+1]=y;tex[t+2]=z;tex[t+3]=u16[i*18+16];
   tex[t+4]=C[0];tex[t+5]=C[1];tex[t+6]=C[2];tex[t+7]=u8[by+28]/255;
   tex[t+8]=C[3];tex[t+9]=C[4];tex[t+10]=C[5];tex[t+11]=u8[by+29]/255;
   tex[t+12]=u8[by+30]/255;tex[t+13]=u8[by+31]/255;
  }
  postMessage({type:'tex',tex},[tex.buffer]);
 } else if(d.type=='sort'){
  const m=d.mvp;let mn=1e30,mx=-1e30;
  for(let i=0;i<N;i++){const z=m[2]*cx[i*3]+m[6]*cx[i*3+1]+m[10]*cx[i*3+2];depth[i]=z;if(z<mn)mn=z;if(z>mx)mx=z}
  const K=65536,counts=new Uint32Array(K),s=(K-1)/(mx-mn+1e-9);
  for(let i=0;i<N;i++){const v=((depth[i]-mn)*s)|0;qb[i]=v;counts[v]++}
  const starts=new Uint32Array(K);for(let k=1;k<K;k++)starts[k]=starts[k-1]+counts[k-1];
  const order=new Uint32Array(N);for(let i=0;i<N;i++)order[starts[qb[i]]++]=i;
  postMessage({type:'order',order},[order.buffer]);
 }}`;
const worker=new Worker(URL.createObjectURL(new Blob([wsrc],{type:"text/javascript"})));
// ---- GL ----
const cv=document.getElementById("c"),gl=cv.getContext("webgl2",{antialias:false});
const VS=`#version 300 es
precision highp float;precision highp int;
layout(location=0) in vec2 corner;layout(location=1) in uint aIndex;
uniform sampler2D uTex;uniform sampler2D uState;
uniform mat4 uView,uProj;uniform vec2 uFocal,uViewport;
uniform float uSel;uniform int uIso,uDimNamed,uHideBg;
out vec3 vColor;out float vAlpha;out vec2 vPos;
ivec2 tc(uint t){return ivec2(int(t&4095u),int(t>>12u));}
void cull(){gl_Position=vec4(0.,0.,2.,1.);}
void main(){
 uint b=aIndex*4u;
 vec4 t0=texelFetch(uTex,tc(b),0);
 float lab=t0.w;float aMul=1.0;
 if(lab<0.5){ if(uHideBg==1){cull();return;} }
 else{
  vec4 st=texelFetch(uState,ivec2(int(lab),0),0);
  if(st.r<0.5){cull();return;}
  bool sv=abs(lab-uSel)<0.5;
  if(uIso==1&&!sv)aMul*=0.05;
  else if(uDimNamed==1&&st.g>0.5&&!sv)aMul*=0.15;
 }
 vec4 cam=uView*vec4(t0.xyz,1.);
 vec4 p2=uProj*cam;
 float clip=1.2*p2.w;
 if(p2.z<-clip||p2.x<-clip||p2.x>clip||p2.y<-clip||p2.y>clip){cull();return;}
 vec4 t1=texelFetch(uTex,tc(b+1u),0);
 vec4 t2=texelFetch(uTex,tc(b+2u),0);
 vec4 t3=texelFetch(uTex,tc(b+3u),0);
 mat3 Vrk=mat3(t1.x,t1.y,t1.z, t1.y,t2.x,t2.y, t1.z,t2.y,t2.z);
 mat3 J=mat3(uFocal.x/cam.z,0.,-(uFocal.x*cam.x)/(cam.z*cam.z),
             0.,-uFocal.y/cam.z,(uFocal.y*cam.y)/(cam.z*cam.z),
             0.,0.,0.);
 mat3 T=transpose(mat3(uView))*J;
 mat3 c2=transpose(T)*Vrk*T;
 c2[0][0]+=0.3;c2[1][1]+=0.3;
 float mid=(c2[0][0]+c2[1][1])/2.;
 float rad=length(vec2((c2[0][0]-c2[1][1])/2.,c2[0][1]));
 float l1=mid+rad,l2=mid-rad;
 if(l2<0.){cull();return;}
 vec2 dv=vec2(c2[0][1],l1-c2[0][0]);
 vec2 dg=length(dv)>1e-9?normalize(dv):vec2(1.,0.);
 vec2 maj=min(sqrt(2.*l1),1024.)*dg;
 vec2 mnr=min(sqrt(2.*l2),1024.)*vec2(dg.y,-dg.x);
 vec3 col=vec3(t1.w,t2.w,t3.x);
 if(uSel>0.5&&abs(lab-uSel)<0.5)col=mix(col,vec3(0.16,1.,0.84),0.4);
 vColor=col;vAlpha=t3.y*aMul;vPos=corner;
 gl_Position=vec4(p2.xy/p2.w+corner.x*maj/uViewport+corner.y*mnr/uViewport,0.,1.);
}`;
const FS=`#version 300 es
precision highp float;
in vec3 vColor;in float vAlpha;in vec2 vPos;out vec4 frag;
void main(){float A=-dot(vPos,vPos);if(A<-4.)discard;float B=exp(A)*vAlpha;frag=vec4(B*vColor,B);}`;
function sh(t,s){const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);
 if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(o));return o}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,VS));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(pr);if(!gl.getProgramParameter(pr,gl.LINK_STATUS))console.error(gl.getProgramInfoLog(pr));
gl.useProgram(pr);
const qb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,qb);
gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-2,-2,2,-2,-2,2,2,2]),gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
const ib=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,ib);
gl.enableVertexAttribArray(1);gl.vertexAttribIPointer(1,1,gl.UNSIGNED_INT,0,0);gl.vertexAttribDivisor(1,1);
const texData=gl.createTexture();gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texData);
gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
const texState=gl.createTexture();gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,texState);
gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,1024,1,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
gl.uniform1i(gl.getUniformLocation(pr,"uTex"),0);
gl.uniform1i(gl.getUniformLocation(pr,"uState"),1);
const U={};for(const n of["uView","uProj","uFocal","uViewport","uSel","uIso","uDimNamed","uHideBg"])U[n]=gl.getUniformLocation(pr,n);
gl.disable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.ONE_MINUS_DST_ALPHA,gl.ONE);
gl.clearColor(0,0,0,0);
// ---- 相机 ----
let tgt=[0,0,1],th=-1.1,ph=0.9,rad=8;
(function(){let sx=0,sy=0,sz=0,n=0;for(let i=0;i<NP*3;i+=Math.max(3,(NP/5000|0)*3)){sx+=ppos[i];sy+=ppos[i+1];sz+=ppos[i+2];n++}tgt=[sx/n,sy/n,sz/n]})();
function sub3(a,b){return[a[0]-b[0],a[1]-b[1],a[2]-b[2]]}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function norm3(a){const l=Math.hypot(a[0],a[1],a[2]);return[a[0]/l,a[1]/l,a[2]/l]}
function mul44(a,b){const o=new Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s}return o}
function cams(){
 const eye=[tgt[0]+rad*Math.cos(ph)*Math.cos(th),tgt[1]+rad*Math.cos(ph)*Math.sin(th),tgt[2]+rad*Math.sin(ph)];
 const f=norm3(sub3(tgt,eye)),r=norm3(cross(f,[0,0,1])),u=cross(r,f);
 const V=[r[0],u[0],-f[0],0,r[1],u[1],-f[1],0,r[2],u[2],-f[2],0,-dot(r,eye),-dot(u,eye),dot(f,eye),1];
 const a=cv.width/cv.height,fy=1.6,fx=fy/a,zn=0.05,zf=200;
 const P=[fx,0,0,0,0,fy,0,0,0,0,(zf+zn)/(zn-zf),-1,0,0,2*zf*zn/(zn-zf),0];
 return {V,P,MVP:mul44(P,V)}}
let cam=null,dirty=true,needSort=true,sortBusy=false,ordered=false;
worker.onmessage=e=>{
 const d=e.data;
 if(d.type=='tex'){
  gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texData);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,TW,TH,0,gl.RGBA,gl.FLOAT,new Float32Array(d.tex));
  document.getElementById("load").remove();needSort=true;dirty=true;
 } else if(d.type=='order'){
  gl.bindBuffer(gl.ARRAY_BUFFER,ib);gl.bufferData(gl.ARRAY_BUFFER,d.order,gl.DYNAMIC_DRAW);
  sortBusy=false;ordered=true;dirty=true;
 }};
worker.postMessage({type:'init',buf:sbuf,n:NS,tw:TW,th:TH},[sbuf]);
function draw(){
 if(cv.width!==cv.clientWidth*devicePixelRatio||cv.height!==cv.clientHeight*devicePixelRatio){
  cv.width=cv.clientWidth*devicePixelRatio;cv.height=cv.clientHeight*devicePixelRatio;gl.viewport(0,0,cv.width,cv.height)}
 cam=cams();
 gl.uniformMatrix4fv(U.uView,false,new Float32Array(cam.V));
 gl.uniformMatrix4fv(U.uProj,false,new Float32Array(cam.P));
 gl.uniform2f(U.uFocal,cam.P[0]*cv.width/2,cam.P[5]*cv.height/2);
 gl.uniform2f(U.uViewport,cv.width,cv.height);
 gl.clear(gl.COLOR_BUFFER_BIT);
 if(ordered)gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,NS);
 dirty=false}
(function loop(){
 if(needSort&&!sortBusy&&cam){sortBusy=true;needSort=false;
  worker.postMessage({type:'sort',mvp:new Float32Array(cam.MVP)})}
 if(dirty)draw();requestAnimationFrame(loop)})();
// ---- 状态/选择 ----
let sel=null,iso=false;
function statePass(){
 const fmin=+document.getElementById("fMin").value/100,fmax=+document.getElementById("fMax").value/100;
 document.getElementById("fLab").textContent=`${(fmin*100)|0}cm – ${fmax>=4?"∞":((fmax*100)|0)+"cm"}`;
 const buf=new Uint8Array(1024*4);
 for(const r of META.ranges){
  if(r.id<0)continue;
  const hid=r.diag<fmin||(fmax<4&&r.diag>fmax);
  vis[r.idx]=hid?0:1;
  buf[r.idx*4]=hid?0:255;
  buf[r.idx*4+1]=names[String(r.id)]?255:0;
 }
 gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,texState);
 gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,1024,1,gl.RGBA,gl.UNSIGNED_BYTE,buf);
 gl.uniform1f(U.uSel,sel?sel.idx:0);
 gl.uniform1i(U.uIso,iso?1:0);
 gl.uniform1i(U.uDimNamed,document.getElementById("dimNamed").checked?1:0);
 gl.uniform1i(U.uHideBg,document.getElementById("bg").checked?0:1);
 dirty=true}
function updStats(){const t=KMAX,d=Object.keys(names).filter(k=>byId[+k]).length;
 document.getElementById("stats").textContent=`已命名 ${d}/${t}`}
function pick(mx,my){
 const w=cv.clientWidth,h=cv.clientHeight,M=cam.MVP;let best=1e9,bi=-1;
 for(const r of META.ranges){if(r.id<0||!vis[r.idx])continue;
  const step=Math.max(1,(r.n/220)|0);
  for(let i=r.s;i<r.s+r.n;i+=step){
   const x=ppos[i*3],y=ppos[i*3+1],z=ppos[i*3+2];
   const cw=M[3]*x+M[7]*y+M[11]*z+M[15];if(cw<=0)continue;
   const sx=((M[0]*x+M[4]*y+M[8]*z+M[12])/cw*0.5+0.5)*w;
   const sy=(1-((M[1]*x+M[5]*y+M[9]*z+M[13])/cw*0.5+0.5))*h;
   const d2=(sx-mx)**2+(sy-my)**2;if(d2<best){best=d2;bi=r.id}}}
 return best<20*20?bi:-1}
function select(id){
 sel=id>=0?byId[id]:null;iso=false;statePass();
 const s=document.getElementById("sel"),tb=document.getElementById("thumbs");
 if(!sel){s.innerHTML="点击一个物体选中实例";tb.innerHTML="";return}
 const nm=names[String(sel.id)]||"";
 s.innerHTML=`<b>#${sel.id}</b> ${nm?"= <b>"+nm+"</b>":"<i>(未命名)</i>"}<br>`+
  `高斯 <b>${sel.ng}</b> · 视角 <b>${sel.nv}</b> · 对角 <b>${(sel.diag*100).toFixed(0)}cm</b><br>`+
  `中心 (${sel.c[0]}, ${sel.c[1]}, ${sel.c[2]}) z=${sel.c[2]}`;
 document.getElementById("nameIn").value=nm;document.getElementById("nameIn").focus();
 tb.innerHTML=`<img src="thumbs/inst_${sel.id}_0.jpg" onerror="this.remove()">`+
              `<img src="thumbs/inst_${sel.id}_1.jpg" onerror="this.remove()">`}
function renderNamed(){
 document.getElementById("named").innerHTML=Object.entries(names).sort((a,b)=>+a[0]-+b[0])
  .map(([k,v])=>`<div>#${k} = ${v}</div>`).join("");updStats()}
// ---- 输入 ----
let drag=0,px=0,py=0,downXY=null;
cv.addEventListener("mousedown",e=>{drag=e.button===0?1:2;px=e.clientX;py=e.clientY});
addEventListener("mouseup",()=>drag=0);
addEventListener("mousemove",e=>{if(!drag)return;const dx=e.clientX-px,dy=e.clientY-py;px=e.clientX;py=e.clientY;
 if(drag===1){th-=dx*0.006;ph=Math.min(1.5,Math.max(-1.5,ph+dy*0.006))}
 else{const s=rad*0.0016;tgt[0]+=Math.sin(th)*dx*s;tgt[1]-=Math.cos(th)*dx*s;tgt[2]+=dy*s}
 needSort=true;dirty=true});
cv.addEventListener("wheel",e=>{rad*=e.deltaY>0?1.12:0.89;rad=Math.min(60,Math.max(0.4,rad));needSort=true;dirty=true;e.preventDefault()},{passive:false});
cv.addEventListener("contextmenu",e=>e.preventDefault());
cv.addEventListener("pointerdown",e=>downXY=[e.clientX,e.clientY]);
cv.addEventListener("pointerup",e=>{if(!downXY)return;
 if(Math.hypot(e.clientX-downXY[0],e.clientY-downXY[1])<4&&e.button===0)select(pick(e.clientX,e.clientY));downXY=null});
addEventListener("keydown",e=>{
 if(e.target.tagName==="INPUT"&&e.key!=="Escape")return;
 if(e.key==="i"||e.key==="I"){iso=!iso;statePass()}
 if(e.key==="Escape")select(-1)});
document.getElementById("nameIn").addEventListener("keydown",e=>{
 if(e.key!=="Enter"||!sel)return;const v=e.target.value.trim();
 if(v)names[String(sel.id)]=v;else delete names[String(sel.id)];
 renderNamed();select(sel.id)});
document.getElementById("jump").addEventListener("keydown",e=>{
 if(e.key!=="Enter")return;const id=parseInt(e.target.value);if(byId[id])select(id)});
document.getElementById("next").onclick=()=>{
 const un=META.ranges.filter(r=>r.id>=0&&!names[String(r.id)]&&vis[r.idx]).sort((a,b)=>b.diag-a.diag);
 if(un.length)select(un[0].id)};
document.getElementById("iso").onclick=()=>{iso=!iso;statePass()};
document.getElementById("exp").onclick=()=>{
 const blob=new Blob([JSON.stringify(names,null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="names.json";a.click()};
for(const id of["bg","dimNamed","fMin","fMax"])document.getElementById(id).addEventListener("input",statePass);
statePass();renderNamed();
</script></body></html>
"""

if __name__ == "__main__":
    main()
