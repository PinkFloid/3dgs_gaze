"""Real scene renders and an explicitly illustrative v2 query for the method figure."""
from pathlib import Path
import sys, json, math, os, argparse

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,default=os.environ.get('GAZESPLAT_CKPT'),
                    help='Trained splatfacto checkpoint; may also be set by GAZESPLAT_CKPT.')
parser.add_argument('--segmentation',type=Path,default=ROOT/'SceneRebuild/lab_result/segmentation_sam')
args=parser.parse_args()
if args.checkpoint is None:
    parser.error('Pass --checkpoint /path/to/step-000029999.ckpt or set GAZESPLAT_CKPT.')
if not args.checkpoint.is_file():
    parser.error(f'Checkpoint not found: {args.checkpoint}')

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

OUT=ROOT/'output/fig2_complete/assets'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'Eye_Tracker/tools'))
from gaze_to_world import SplatDepth
from gaze_object import cone_votes, rank_votes, object_radii_by_name, pooled_centroids_by_name

CKPT=args.checkpoint
SEG=args.segmentation
sd=SplatDepth(CKPT)
z=np.load(SEG/'points.npz'); xyz=z['xyz']; labels=z['label']
names=json.loads((SEG/'names.json').read_text(encoding='utf-8'))
meta=json.loads((SEG/'instances.json').read_text(encoding='utf-8'))
centroids=pooled_centroids_by_name(meta['instances'],names)
places=set(json.loads((SEG/'places.json').read_text(encoding='utf-8')))
targets=set(n for n in names.values() if n)-places
radii=object_radii_by_name(xyz,labels,names,only=targets)
tree=cKDTree(xyz)
PAL={'球R':(0,135,122),'球M':(126,142,159),'球L':(162,171,182)}
ENAME={'球R':'ball_R','球M':'ball_M','球L':'ball_L','白杯1':'cup_1','白杯2':'cup_2','苹果粉':'apple_pink','苹果红':'apple_red','香蕉':'banana','橘子':'orange','红杯':'cup_red','物品台':'cart','纸箱子':'box'}

def cam(eye,at,W,H,hfov):
    eye=np.asarray(eye,float); at=np.asarray(at,float)
    z=(at-eye);z/=np.linalg.norm(z)
    x=np.cross(z,[0.,0.,1.]);x/=np.linalg.norm(x);y=np.cross(z,x)
    w2c=np.eye(4);w2c[:3,:3]=np.stack([x,y,z]);w2c[:3,3]=-w2c[:3,:3]@eye
    f=W/2/np.tan(np.radians(hfov)/2)
    K=np.array([[f,0,W/2],[0,f,H/2],[0,0,1.]])
    return w2c,K

def render(w2c,K,W,H,subset=None,colors=None,sh=3):
    args=[sd.means,sd.quats,sd.scales,sd.opac,sd.colors if colors is None else colors]
    if subset is not None:args=[a[subset] for a in args]
    with torch.no_grad():
        out,alpha,_=sd.rasterization(*args,torch.tensor(w2c,dtype=torch.float32,device='cuda')[None],
                  torch.tensor(K,dtype=torch.float32,device='cuda')[None],W,H,sh_degree=sh,render_mode='RGB+ED',rasterize_mode='classic')
    return out[0].cpu().numpy(),alpha[0,...,0].cpu().numpy()

def save_rgb(name,rgb):
    cv2.imwrite(str(OUT/name),cv2.cvtColor(np.clip(rgb*255,0,255).astype(np.uint8),cv2.COLOR_RGB2BGR))

means=sd.means.detach().cpu().numpy()
crop=(means[:,0]>.57)&(means[:,0]<1.58)&(means[:,1]>.11)&(means[:,1]<.88)&(means[:,2]>.29)&(means[:,2]<1.12)
subset=torch.as_tensor(np.flatnonzero(crop),device='cuda')
print('clipped scene gaussians',len(subset),flush=True)
mtree=cKDTree(means)
parts={}
for name in PAL:
    labs=[int(k) for k,v in names.items() if v==name]
    d,idx=mtree.query(xyz[np.isin(labels,labs)])
    parts[name]=torch.as_tensor(np.unique(idx[d<.012]),device='cuda')

metadata={'map_version':'v9','checkpoint':str(CKPT),'hero':{},'query_note':'Illustrative synthetic fixation in the real map; not a recorded user trial.'}
for vi,(eye,at,fov) in enumerate([
    ([2.35,-1.35,1.85],[1.05,.46,.64],35),
    ([-.25,-1.3,1.8],[1.05,.46,.64],35),
    ([2.0,-.95,1.9],[1.05,.46,.66],38)
]):
    W,H=1600,920
    w2c,K=cam(eye,at,W,H,fov)
    im,a=render(w2c,K,W,H,subset)
    rgb=np.clip(im[...,:3]+1-a[...,None],0,1)
    save_rgb(f'hero_{vi}_raw.png',rgb)
    objects={}
    for name,idx in parts.items():
        oi,oa=render(w2c,K,W,H,idx)
        vis=(oa>.2)&(oi[...,3]<im[...,3]+.03)
        weight=(.5*np.clip(oa,0,1)*vis)[...,None]
        rgb=rgb*(1-weight)+np.array(PAL[name])/255*weight
        mask=((oa>.5)&vis).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        rgb8=np.clip(rgb*255,0,255).astype(np.uint8)
        cv2.drawContours(rgb8,[c for c in contours if cv2.contourArea(c)>15],-1,PAL[name],3,cv2.LINE_AA)
        rgb=rgb8/255.
        pts=np.concatenate([np.asarray(centroids[name]),[1.]])
        pc=w2c@pts;uv=K@pc[:3];uv=uv[:2]/uv[2]
        objects[ENAME[name]]={'uv':uv.tolist(),'contours':[c.reshape(-1,2).tolist() for c in contours if cv2.contourArea(c)>15]}
    save_rgb(f'hero_{vi}.png',rgb)
    metadata['hero'][str(vi)]={'eye':eye,'w2c':w2c.tolist(),'K':K.tolist(),'size':[W,H],'objects':objects}
    print('hero',vi,'done',flush=True)

# One geometrically reproducible model query, deliberately not labelled as eye-tracking data.
origin=np.array([.88,-2.25,1.65])
target=np.asarray(centroids['球R'])
# Center slightly between the two rightmost balls so competitors remain visible.
pt=target+np.array([-.008,0,0])
sigma=math.radians(1.0)
S=65
votes,kern=cone_votes(sd,tree,labels,origin,pt,sigma,2.,S,.05)
def name_of(l):return names.get(str(l),'') or {0:'floor',1:'ceiling',2:'wall',3:'wall',4:'wall',5:'wall'}.get(l,f'object#{l}')
rank=rank_votes(votes,kern,name_of,targets,centroids,radii,sigma,float(np.linalg.norm(pt-origin)))
depth,alpha,dirs,tmul=sd.patch_along_ray(origin,(pt-origin)/np.linalg.norm(pt-origin),2*sigma,S)
X=origin+(depth*tmul)[...,None]*dirs
dd,ii=tree.query(X.reshape(-1,3),distance_upper_bound=.05)
lab=np.full((S*S),-1);valid=np.isfinite(dd)
lab[valid]=labels[ii[valid]];lab=lab.reshape(S,S)
lab_img=np.full((S,S,3),[235,238,240],dtype=np.uint8)
for k,v in names.items():
    color={'球R':(0,135,122),'白杯1':(140,139,184),'红杯':(187,146,120)}.get(v,(215,221,225))
    lab_img[lab==int(k)]=color
lab_img[(depth<=.05)|(depth>=12)|(depth*tmul>=np.linalg.norm(pt-origin)+.5)]=[246,247,248]
# Fade invalid low-opacity surfaces to white consistently with rendered support.
lab_img=(lab_img*np.clip(alpha,0,1)[...,None]+255*(1-np.clip(alpha,0,1)[...,None])).astype(np.uint8)
lab_big=cv2.resize(lab_img,(780,780),interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(OUT/'query_instances.png'),cv2.cvtColor(lab_big,cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUT/'query_alpha.png'),cv2.resize((np.clip(alpha,0,1)*255).astype(np.uint8),(780,780),interpolation=cv2.INTER_NEAREST))
lo,hi=np.percentile(depth[depth>.05],[3,97]);dn=np.clip((depth-lo)/max(hi-lo,.001),0,1)
dc=cv2.applyColorMap((dn*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS)
cv2.imwrite(str(OUT/'query_depth.png'),cv2.resize(dc,(780,780),interpolation=cv2.INTER_NEAREST))
metadata['query']={'origin':origin.tolist(),'point':pt.tolist(),'sigma_deg':1.,'span_sigmas':2.,'patch':S,'rank':rank,'depth_limits':[float(lo),float(hi)]}
# A real novel-view RGB image of the same map, for a clean map-view inset if needed.
w2c,K=cam(origin,target,1200,720,22)
im,a=render(w2c,K,1200,720)
save_rgb('query_rgb.png',np.clip(im[...,:3]+1-a[...,None],0,1))
(OUT/'render_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print('query',json.dumps(rank,ensure_ascii=False),flush=True)
