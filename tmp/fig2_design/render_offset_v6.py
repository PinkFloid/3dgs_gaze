"""Recorded low gaze: full origin, physical query support, and visible ray hits."""
from pathlib import Path
import sys,json,math,itertools
import numpy as np
import cv2
import torch
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Eye_Tracker/tools'))
from gaze_to_world import SplatDepth
from gaze_object import cone_votes,rank_votes,pooled_centroids_by_name,object_radii_by_name
from render_recorded_assets import camera
OUT=ROOT/'output/fig2_offset_v6/assets';OUT.mkdir(parents=True,exist_ok=True)
sel=next(r for r in json.loads((OUT/'candidates.json').read_text()) if abs(r['event']['t_start']-14000.495956)<.001)
e=sel['event'];o=np.array(e['origin_world']);p=np.array(e['centroid_world']);D=np.linalg.norm(p-o);d=(p-o)/D
x=np.cross(d,[0,0,1]);x/=np.linalg.norm(x);y=np.cross(d,x)
sigma=math.radians(1);half=2*sigma
far=np.array([o+D*(d+math.tan(half)*(sx*x+sy*y)) for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]])
SEG=ROOT/'SceneRebuild/lab_result/segmentation_sam'
names=json.loads((SEG/'names.json').read_text());inst=json.loads((SEG/'instances.json').read_text())['instances']
z=np.load(SEG/'points.npz');tree=cKDTree(z['xyz']);labels=z['label']
targets=set(filter(None,names.values()))-set(json.loads((SEG/'places.json').read_text()))
centers=pooled_centroids_by_name(inst,names);radii=object_radii_by_name(z['xyz'],labels,names,only=targets)
sd=SplatDepth(ROOT/'SceneRebuild/lab_result/splatfacto/2026-08-20_201525_nobottle/nerfstudio_models/step-000029999.ckpt')
v,k=cone_votes(sd,tree,labels,o,p,sigma,2,33,.05)
rank=rank_votes(v,k,lambda l:names.get(str(l),'') or str(l),targets,centers,radii,sigma,D)
dep,alpha,dirs,tmul=sd.patch_along_ray(o,d,half,33);hits=o+(dep*tmul)[...,None]*dirs
dd,idx=tree.query(hits.reshape(-1,3),distance_upper_bound=.05)
valid=np.isfinite(dd).reshape(33,33)&(dep>.05)&(dep<12)&(dep*tmul<D+.5)
labs=np.full(1089,-1);ok=np.isfinite(dd);labs[ok]=labels[idx[ok]];labs=labs.reshape(33,33)
w=k['w'].reshape(33,33);evidence=w*alpha
pooled={n:sum(m for l,m in v.items() if names.get(str(l))==n) for n in targets}
T=sum(pooled.values());shares={n:m/T for n,m in pooled.items()}
target_mask=valid&np.isin(labs,[int(l) for l,n in names.items() if n in targets])
grid=np.zeros((33,33),bool);grid[::4,::4]=True
ray_mask=grid&target_mask
selected=np.argwhere(ray_mask)
print('full shares',json.dumps(shares,ensure_ascii=False),flush=True)
print('display rays',[(int(l),int((labs[ray_mask]==l).sum())) for l in np.unique(labs[ray_mask])],flush=True)
np.savez_compressed(OUT/'query_arrays.npz',points=hits,labels=labs,valid=valid,alpha=alpha,weights=w,evidence=evidence,display_ray_mask=ray_mask)
EN={'球L':'ball_L','球M':'ball_M','球R':'ball_R'}
edges=[(a,b) for a in range(8) for b in range(a+1,8) if (a^b) in (1,2,4)]
boxes={}
for n in EN:
    parts=[r for r in inst if names.get(str(r['id']))==n]
    lo=np.min([r['bbox_min'] for r in parts],0);hi=np.max([r['bbox_max'] for r in parts],0)
    boxes[EN[n]]={'lo':lo.tolist(),'hi':hi.tolist(),'corners':list(itertools.product(*zip(lo,hi)))}
means=sd.means.cpu().numpy()
crop=(means[:,0]>.57)&(means[:,0]<1.58)&(means[:,1]>.11)&(means[:,1]<.88)&(means[:,2]>.68)&(means[:,2]<1.12)
ii=torch.tensor(np.flatnonzero(crop),device=sd.dev)
vals=[a[ii] for a in [sd.means,sd.quats,sd.scales,sd.opac,sd.colors]]
eyes=[[1.4,-3.7,2.2],[1.7,-5,2.9],[1.7,-5,2.3],[1.7,-6,2.7],
      [2.3,-5,2.8],[1.4,-4.3,2.1],[1.4,-4.5,2.7],[1.9,-4,2.9]]
AMBER=np.array([188.,126.,39.]);TEAL=np.array([0.,141.,123.]);GRAY=np.array([117.,133.,147.])
meta={'selected':sel,'recording':'2026_08_27/019','query':{'sigma_deg':1,'span_sigmas':2,'patch':33,'distance_m':float(D),'section_side_m':float(2*D*math.tan(half)),'rank':rank,'object_shares':shares,'object_evidence':pooled,'W':k['W']},
      'origin':o.tolist(),'axis_endpoint':p.tolist(),'central_surface_point':hits[16,16].tolist(),'frustum_corners':far.tolist(),
      'boxes':boxes,'box_edges':edges,'views':{},'display_ray_grid':'Rows and columns 0,4,...,32 of the 33x33 patch; rays with valid registered-object matches only. Opacity reflects angular weight times opacity.',
      'display_rays':[{'row':int(r),'col':int(c),'name':names.get(str(labs[r,c]),''),'point':hits[r,c].tolist(),'weight':float(evidence[r,c])} for r,c in selected]}
for vi,eye in enumerate(eyes):
    W,H=1800,850
    mat,K=camera(eye,[1.0,.1,.90],W,H,30)
    # Fit one perspective camera to the recorded origin and the table bounds.
    fit=np.array([o]+list(itertools.product([.61,1.57],[.13,.86],[.70,1.10])))
    q=np.c_[fit,np.ones(len(fit))]@mat.T;xy=q[:,:2]/q[:,2:3]
    lo=xy.min(0);hi=xy.max(0);f=min((W-190)/(hi[0]-lo[0]),(H-180)/(hi[1]-lo[1]))
    K=np.array([[f,0,(W-f*(lo[0]+hi[0]))/2],[0,f,(H-f*(lo[1]+hi[1]))/2],[0,0,1.]])
    with torch.no_grad():
        im,al,_=sd.rasterization(*vals,torch.tensor(mat,dtype=torch.float32,device=sd.dev)[None],torch.tensor(K,dtype=torch.float32,device=sd.dev)[None],W,H,sh_degree=3,render_mode='RGB+ED',rasterize_mode='classic')
    im=im[0].cpu().numpy();al=al[0,...,0].cpu().numpy();rgb=np.clip(im[...,:3]+1-al[...,None],0,1)*255
    def project(pts):
        pts=np.atleast_2d(pts);q=np.c_[pts,np.ones(len(pts))]@mat.T;u=q[:,:3]@K.T
        return u[:,:2]/u[:,2:3],q[:,2]
    def cloud(pts,color,opacity,radius=1):
        uv,zz=project(pts);uv=np.round(uv).astype(int)
        a=np.broadcast_to(opacity,len(uv));colors=np.broadcast_to(color,(len(uv),3))
        for (u,v),depth,op,col in zip(uv,zz,a,colors):
            if op<.004 or depth<=0 or not (radius<=u<W-radius and radius<=v<H-radius):continue
            if al[v,u]>.5 and depth>im[v,u,3]+.015:continue
            block=rgb[v-radius:v+radius+1,u-radius:u+radius+1];block[:]=block*(1-op)+col*op
    def edge(a,b,color,opacity=.5,radius=1):
        uv,_=project([a,b]);N=min(5000,max(3,int(np.linalg.norm(uv[1]-uv[0])*1.7)))
        uv,zz=project(a+(b-a)*np.linspace(0,1,N)[:,None]);uv=np.round(uv).astype(int)
        ok=(zz>0)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
        uv=uv[ok];zz=zz[ok]
        ok=(al[uv[:,1],uv[:,0]]<=.5)|(zz<=im[uv[:,1],uv[:,0],3]+.015)
        uv=uv[ok];mask=np.zeros((H,W),np.uint8);mask[uv[:,1],uv[:,0]]=1
        if radius:mask=cv2.dilate(mask,np.ones((2*radius+1,2*radius+1),np.uint8))
        use=mask.astype(bool);rgb[use]=rgb[use]*(1-opacity)+color*opacity
    # Domain corners retain the physical square angular support; no widened cone.
    for ff in far:edge(o,ff,AMBER,.48,1)
    for j in range(4):edge(far[j],far[(j+1)%4],AMBER,.65,1)
    # Draw weaker competing-object rays before the stronger green target rays.
    for r,c in sorted(selected,key=lambda rc:int(labs[tuple(rc)]==261)):
        col=TEAL if labs[r,c]==261 else GRAY
        edge(o,hits[r,c],col,.16+.64*evidence[r,c],1)
    # All matching object samples are shown; no cherry-picked hit positions.
    pp=hits[target_mask];cols=np.repeat(GRAY[None],len(pp),0);cols[labs[target_mask]==261]=TEAL
    cloud(pp,cols,.3+.6*evidence[target_mask],2)
    for name,b in boxes.items():
        col=TEAL if name=='ball_M' else GRAY
        for a,b in edges:edge(np.array(boxes[name]['corners'][a]),np.array(boxes[name]['corners'][b]),col,.8,1)
    end=hits[16,16]
    for a in np.arange(0,1,.028):edge(o+a*(end-o),o+min(a+.014,1)*(end-o),AMBER,.82,1)
    # Cross and source circle are added as editable vectors in the layout.
    proj={'origin':project(o)[0][0].tolist(),'axis_endpoint':project(p)[0][0].tolist(),'central_surface_point':project(end)[0][0].tolist(),
          'corners':project(far)[0].tolist(),'boxes':{n:project(b['corners'])[0].tolist() for n,b in boxes.items()}}
    meta['views'][str(vi)]={'eye':eye,'w2c':mat.tolist(),'K':K.tolist(),'size':[W,H],'projection':proj}
    cv2.imwrite(str(OUT/f'full_{vi}.png'),cv2.cvtColor(np.uint8(np.clip(rgb,0,255)),cv2.COLOR_RGB2BGR))
    print('view',vi,'origin',proj['origin'],'axis',proj['central_surface_point'],flush=True)
(OUT/'geometry.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
