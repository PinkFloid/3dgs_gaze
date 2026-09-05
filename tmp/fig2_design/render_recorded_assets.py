"""Re-render Fig. 2 assets from a logged fixation and the bottle-free v9 map.

This is an offline v2 replay of a historical event, not a claim about the
parameters used during the original online trial. No gaze recentering is used.
"""
from pathlib import Path
import sys, json, math, argparse
import numpy as np
import cv2
import torch
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Eye_Tracker/tools'))
from gaze_to_world import SplatDepth
from gaze_object import cone_votes, rank_votes, object_radii_by_name, pooled_centroids_by_name

OUT=ROOT/'output/fig2_recorded_v3/assets'
SEG=ROOT/'SceneRebuild/lab_result/segmentation_sam'
PAL={'球M':(0,135,122),'球L':(137,148,157),'球R':(137,148,157),
     '香蕉':(190,142,69),'白杯1':(140,139,184),'苹果红':(158,86,111)}
EN={'球M':'ball_M','球L':'ball_L','球R':'ball_R','香蕉':'banana','白杯1':'cup_1','苹果红':'apple_red'}

def camera(eye,at,w,h,hfov):
    eye=np.asarray(eye,float); z=np.asarray(at,float)-eye; z/=np.linalg.norm(z)
    x=np.cross(z,[0.,0.,1.]); x/=np.linalg.norm(x); y=np.cross(z,x)
    mat=np.eye(4);mat[:3,:3]=np.stack([x,y,z]);mat[:3,3]=-mat[:3,:3]@eye
    f=w/2/np.tan(np.radians(hfov)/2)
    return mat,np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])

def save(name,rgb):
    cv2.imwrite(str(OUT/name),cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255),cv2.COLOR_RGB2BGR))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'SceneRebuild/lab_result/splatfacto/2026-08-20_201525_nobottle/nerfstudio_models/step-000029999.ckpt')
    a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    log=ROOT/'Intension/logs/20260827-175225/gaze.jsonl'
    events=[json.loads(l) for l in log.read_text().splitlines()]
    event=next(r for r in events if abs(r['t_start']-14055.8847585)<1e-5 and not r.get('provisional'))
    origin=np.array(event['origin_world']);point=np.array(event['centroid_world'])
    z=np.load(SEG/'points.npz'); xyz=z['xyz']; labels=z['label']
    names=json.loads((SEG/'names.json').read_text())
    instances=json.loads((SEG/'instances.json').read_text())['instances']
    centers=pooled_centroids_by_name(instances,names)
    places=set(json.loads((SEG/'places.json').read_text()))
    targets=set(filter(None,names.values()))-places
    radii=object_radii_by_name(xyz,labels,names,only=targets)
    tree=cKDTree(xyz); sd=SplatDepth(a.checkpoint)

    def render(mat,K,w,h,idx=None):
        vals=[sd.means,sd.quats,sd.scales,sd.opac,sd.colors]
        if idx is not None: vals=[v[idx] for v in vals]
        with torch.no_grad():
            out,al,_=sd.rasterization(*vals,torch.tensor(mat,dtype=torch.float32,device=sd.dev)[None],
                torch.tensor(K,dtype=torch.float32,device=sd.dev)[None],w,h,sh_degree=3,
                render_mode='RGB+ED',rasterize_mode='classic')
        return out[0].cpu().numpy(),al[0,...,0].cpu().numpy()

    mat,K=camera(origin,point,1400,800,18)
    im,alpha=render(mat,K,1400,800)
    save('fixation_view.png',im[...,:3]+1-alpha[...,None])
    meta={'checkpoint':str(a.checkpoint.relative_to(ROOT)), 'recording':'2026_08_27/020',
          'gaze_log':str(log.relative_to(ROOT)),'recorded_event':event,
          'note':'Offline v2 query replay of an unchanged recorded fixation; historical event segmentation and online v1 scores are not recomputed.',
          'view':{'w2c':mat.tolist(),'K':K.tolist(),'size':[1400,800],'hfov_deg':18}}
    sigma=math.radians(1.);S=33;dist=float(np.linalg.norm(point-origin))
    votes,kern=cone_votes(sd,tree,labels,origin,point,sigma,2.,S,.05)
    rank=rank_votes(votes,kern,lambda l:names.get(str(l),'') or f'component_{l}',targets,centers,radii,sigma,dist)
    depth,alpha,dirs,tmul=sd.patch_along_ray(origin,point-origin,2*sigma,S)
    pts=origin+(depth*tmul)[...,None]*dirs
    dd,idx=tree.query(pts.reshape(-1,3),distance_upper_bound=.05)
    valid=np.isfinite(dd).reshape(S,S)&(depth>.05)&(depth<12)&(depth*tmul<dist+.5)
    lab=np.full(S*S,-1);ok=np.isfinite(dd);lab[ok]=labels[idx[ok]];lab=lab.reshape(S,S)
    patch=np.full((S,S,3),235.,dtype=float)
    for label,name in names.items(): patch[lab==int(label)]=PAL.get(name,(208,217,224))
    patch[~valid]=[248,249,250]
    patch=patch*np.clip(alpha,0,1)[...,None]+255*(1-np.clip(alpha,0,1)[...,None])
    save('query_instances.png',cv2.resize(patch/255.,(792,792),interpolation=cv2.INTER_NEAREST))
    save('query_alpha.png',np.repeat(cv2.resize(alpha,(792,792),interpolation=cv2.INTER_NEAREST)[...,None],3,axis=2))
    lo,hi=np.percentile(depth[valid],[2,98]); dn=np.clip((depth-lo)/(hi-lo),0,1)
    dc=cv2.cvtColor(cv2.applyColorMap(np.uint8(dn*255),cv2.COLORMAP_VIRIDIS),cv2.COLOR_BGR2RGB)
    save('query_depth.png',cv2.resize(dc/255.,(792,792),interpolation=cv2.INTER_NEAREST))
    np.savez_compressed(OUT/'query_arrays.npz',depth=depth,alpha=alpha,valid=valid,labels=lab,weights=kern['w'].reshape(S,S))
    meta['query']={'sigma_deg':1,'half_fov_deg':2,'resolution':S,'distance_m':dist,
                   'rank':rank,'votes':votes,'total_kernel_mass':kern['W']}
    print('REPLAY',json.dumps(rank,ensure_ascii=False),flush=True)

    # Preserve the successful camera and crop of the prior hero, while rendering
    # the map version consistent with the recording and highlighting ball_M.
    means=sd.means.cpu().numpy()
    crop=(means[:,0]>.57)&(means[:,0]<1.58)&(means[:,1]>.11)&(means[:,1]<.88)&(means[:,2]>.29)&(means[:,2]<1.12)
    subset=torch.tensor(np.flatnonzero(crop),device=sd.dev)
    mat,K=camera([-.25,-1.3,1.8],[1.05,.46,.64],1600,920,35)
    im,alpha=render(mat,K,1600,920,subset)
    rgb=np.clip(im[...,:3]+1-alpha[...,None],0,1)
    save('hero_raw.png',rgb)
    mtree=cKDTree(means); objects={}
    for name in ['球L','球M','球R']:
        labs=[int(l) for l,n in names.items() if n==name]
        dd,idx=mtree.query(xyz[np.isin(labels,labs)])
        ids=torch.tensor(np.unique(idx[dd<.012]),device=sd.dev)
        oi,oa=render(mat,K,1600,920,ids)
        visible=(oa>.2)&(oi[...,3]<im[...,3]+.03)
        blend=(.43*oa*visible)[...,None]
        rgb=rgb*(1-blend)+np.array(PAL[name])[None,None,:]/255*blend
        mask=((oa>.5)&visible).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=[c for c in contours if cv2.contourArea(c)>15]
        img8=np.uint8(np.clip(rgb,0,1)*255)
        cv2.drawContours(img8,contours,-1,PAL[name],3,cv2.LINE_AA);rgb=img8/255.
        pc=mat@np.r_[centers[name],1.];uv=K@pc[:3];uv=uv[:2]/uv[2]
        objects[EN[name]]={'uv':uv.tolist(),'contours':[c.reshape(-1,2).tolist() for c in contours]}
    save('hero.png',rgb)
    meta['hero']={'objects':objects,'w2c':mat.tolist(),'K':K.tolist(),'size':[1600,920]}
    (OUT/'provenance.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print(OUT,flush=True)

if __name__=='__main__': main()
