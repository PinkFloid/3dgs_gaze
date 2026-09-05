"""Screen real recorded fixations for a readable low-axis method example."""
from pathlib import Path
import sys,json,math
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Eye_Tracker/tools'))
from gaze_to_world import SplatDepth
from gaze_object import cone_votes,rank_votes,pooled_centroids_by_name,object_radii_by_name
OUT=ROOT/'output/fig2_offset_v6/assets';OUT.mkdir(parents=True,exist_ok=True)
seg=ROOT/'SceneRebuild/lab_result/segmentation_sam'
names=json.loads((seg/'names.json').read_text());inst=json.loads((seg/'instances.json').read_text())['instances']
z=np.load(seg/'points.npz');tree=cKDTree(z['xyz']);labels=z['label']
targets=set(filter(None,names.values()))-set(json.loads((seg/'places.json').read_text()))
centers=pooled_centroids_by_name(inst,names);radii=object_radii_by_name(z['xyz'],labels,names,only=targets)
sd=SplatDepth(ROOT/'SceneRebuild/lab_result/splatfacto/2026-08-20_201525_nobottle/nerfstudio_models/step-000029999.ckpt')
events=[json.loads(l) for l in (ROOT/'Intension/logs/20260827-175225/gaze.jsonl').read_text().splitlines()]
results=[]
for e in events:
    if e.get('provisional') or e.get('object')!='球M' or not 13999<e['t_start']<14070:continue
    o=np.array(e['origin_world']);p=np.array(e['centroid_world']);D=np.linalg.norm(p-o);d=(p-o)/D
    c=np.array(centers['球M']);near=o+np.dot(c-o,d)*d;below=(c[2]-near[2])*100
    if not 2<below<10:continue
    v,k=cone_votes(sd,tree,labels,o,p,math.radians(1),2,33,.05)
    rank=rank_votes(v,k,lambda l:names.get(str(l),'') or str(l),targets,centers,radii,math.radians(1),D)
    dep,alpha,dirs,tmul=sd.patch_along_ray(o,d,math.radians(2),33)
    pts=o+(dep*tmul)[...,None]*dirs
    dd,ix=tree.query(pts.reshape(-1,3),distance_upper_bound=.05)
    valid=(np.isfinite(dd).reshape(33,33))&(dep>.05)&(dep<12)&(dep*tmul<D+.5)
    lab=np.full(1089,-1);lab[np.isfinite(dd)]=labels[ix[np.isfinite(dd)]];lab=lab.reshape(33,33)
    counts={n:int((valid&np.isin(lab,[int(l) for l,nm in names.items() if nm==n])).sum()) for n in targets}
    center_label=int(lab[16,16]) if valid[16,16] else -1
    result={'event':e,'below_cm':below,'distance_m':D,'rank':rank,'counts':counts,
            'center_ray_label':center_label,'center_ray_name':names.get(str(center_label),str(center_label)),
            'center_ray_point':pts[16,16].tolist(),'nearest_axis_to_target':near.tolist()}
    results.append(result)
    print(json.dumps({'t':e['t_start'],'below_cm':round(below,1),'center':result['center_ray_name'],
                      'rank':rank,'counts':{n:c for n,c in counts.items() if c}},ensure_ascii=False),flush=True)
(OUT/'candidates.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
