"""Matched real / RGB render / 50:50 blend / instance-color render.

Only iPhone construction images and their matching aligned map poses are accepted.
Do not substitute later recordings: the cart and tennis balls were moved.
No post-render registration, color matching or image deformation is applied.
"""
from pathlib import Path
import sys,json,colorsys,itertools,argparse
import numpy as np,cv2
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'Eye_Tracker/tools'))
from gaze_to_world import SplatDepth

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--segmentation',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,default=ROOT/'output/scene_map_comparison_iphone/assets')
    ap.add_argument('--transforms',type=Path,required=True,help='Aligned transforms JSON for the exact construction images and map.')
    ap.add_argument('--frame-name',required=True,help='Exact file_path or basename in the transforms file.')
    ap.add_argument('--max-width',type=int,default=2200)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    if a.max_width<=0:ap.error('--max-width must be positive')
    sd=SplatDepth(a.checkpoint);torch=sd.torch
    ins=json.loads((a.segmentation/'instances.json').read_text(encoding='utf-8-sig'));names=json.loads((a.segmentation/'names.json').read_text(encoding='utf-8-sig'));points=np.load(a.segmentation/'points.npz')
    expected_names={'259':'球L','261':'球M','263':'球R','266':'白杯1','267':'白杯2'}
    if len(ins['instances'])!=259 or len(set(filter(None,names.values())))!=13 or any(names.get(k)!=v for k,v in expected_names.items()):
        raise ValueError('Expected the final-layout snapshot with 259 instances, 13 distinct names, and the original ball/cup IDs; do not mix map versions')
    xyz,lab=points['xyz'],points['label'];means=sd.means.cpu().numpy();dd,ix=cKDTree(means).query(xyz)
    if float(dd.max())>1e-6:raise ValueError('Segmentation Gaussian centers do not match the checkpoint.')
    bg={int(k):v for k,v in ins.get('background',{}).items()};byid={int(r['id']):r for r in ins['instances']}
    fixed={259:'#4C78A8',261:'#009F87',263:'#E69F00',266:'#9560BA',267:'#D76891'}
    palette={};colors=np.full((len(means),3),.90,np.float32)
    for n,l in enumerate(sorted(int(v) for v in np.unique(lab) if int(v) not in bg)):
        if l in fixed:rgb=np.array([int(fixed[l][i:i+2],16)/255 for i in [1,3,5]])
        else:rgb=np.array(colorsys.hsv_to_rgb((n*.618033988749895+.13)%1,.62,.85))
        palette[str(l)]={'name':names.get(str(l),''),'hex':'#'+''.join(f'{round(c*255):02X}' for c in rgb),'rgb':rgb.tolist()}
        colors[ix[lab==l]]=rgb
    for l,n in bg.items():colors[ix[lab==l]]={'floor':.77,'ceiling':.94}.get(n,.88)
    col_t=torch.tensor(colors,dtype=torch.float32,device=sd.dev)
    tj=json.loads(a.transforms.read_text(encoding='utf-8-sig'));frames=[f for f in tj['frames'] if f['file_path']==a.frame_name or Path(f['file_path']).name==a.frame_name]
    if len(frames)!=1:raise ValueError('frame-name must identify exactly one transforms frame')
    fr=frames[0];get=lambda k:fr.get(k,tj.get(k));W,H=int(get('w')),int(get('h'))
    if get('camera_model') not in ('OPENCV','PINHOLE'):raise ValueError('Expected OPENCV or PINHOLE camera metadata')
    ip=a.transforms.parent/fr['file_path'];real=cv2.imdecode(np.fromfile(ip,dtype=np.uint8),cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if real is None or real.shape[:2]!=(H,W):raise ValueError('Image size/orientation must match the transforms camera metadata')
    K=np.array([[get('fl_x'),0,get('cx')],[0,get('fl_y'),get('cy')],[0,0,1.]])
    dist=np.array([get('k1') or 0,get('k2') or 0,get('p1') or 0,get('p2') or 0,get('k3') or 0])
    real=cv2.undistort(real,K,dist,None,K)
    T=np.array(fr['transform_matrix']);T[:3,1:3]*=-1 # OpenGL to OpenCV optical axes.
    scale=min(1,a.max_width/W)
    if scale<1:
        new_w,new_h=round(W*scale),round(H*scale)
        real=cv2.resize(real,(new_w,new_h),interpolation=cv2.INTER_AREA);K[0]*=new_w/W;K[1]*=new_h/H
    H,W=real.shape[:2]
    entries=[({'key':'iphone_'+Path(fr['file_path']).stem,'source_type':'iPhone construction image','source_image':str(ip),'transforms':str(a.transforms),'T_world_cam':T.tolist(),'K':K.tolist(),'size':[W,H],'rectification':'OpenCV calibration undistortion; no post-render registration'},real)]
    for source,real in entries:
        folder=a.out_dir/source['key'];folder.mkdir(parents=True,exist_ok=True)
        T=np.array(source['T_world_cam']);mat=np.linalg.inv(T);K=np.array(source['K']);W,H=source['size']
        with torch.no_grad():
            rgb,alpha=sd._render(mat,K,W,H)
            inst,_,_=sd.rasterization(sd.means,sd.quats,sd.scales,sd.opac,col_t,torch.tensor(mat,dtype=torch.float32,device=sd.dev)[None],torch.tensor(K,dtype=torch.float32,device=sd.dev)[None],W,H,sh_degree=None,render_mode='RGB',rasterize_mode='classic',backgrounds=torch.ones(1,3,device=sd.dev))
        render=np.uint8(np.clip(rgb[...,:3]+1-alpha,0,1)*255)[...,::-1]
        instance=np.uint8(np.clip(inst[0].cpu().numpy(),0,1)*255)[...,::-1]
        blend=cv2.addWeighted(real,.5,render,.5,0)
        tint=cv2.addWeighted(render,.60,instance,.40,0)
        for name,im in [('real',real),('render',render),('blend',blend),('instances',instance),('instance_overlay',tint)]:cv2.imencode('.png',im)[1].tofile(folder/(name+'.png'))
        def project(pts):
            pts=np.atleast_2d(pts);q=np.c_[pts,np.ones(len(pts))]@mat.T
            if np.any(q[:,2]<=0):raise ValueError('A target is behind this camera; choose another construction frame')
            u=q[:,:3]@K.T;return u[:,:2]/u[:,2:3]
        objs={}
        for iid in [259,261,263,266,267]:
            r=byid[iid];corners=np.array(list(itertools.product(*zip(r['bbox_min'],r['bbox_max']))));uv=project(corners)
            if np.any(uv<0) or np.any(uv[:,0]>=W) or np.any(uv[:,1]>=H):raise ValueError(f'Target {iid} is outside this view; choose another construction frame')
            objs[str(iid)]={'name':names[str(iid)],'center_uv':project(r['centroid'])[0].tolist(),'bbox_uv':uv.tolist(),'color':palette[str(iid)]['hex']}
        def roi(ids,pad=8,aspect=None):
            uv=np.vstack([objs[str(i)]['bbox_uv'] for i in ids]);lo=uv.min(0)-pad;hi=uv.max(0)+pad;center=(lo+hi)/2;wh=hi-lo
            if aspect:
                wh[0]=max(wh[0],wh[1]*aspect);wh[1]=max(wh[1],wh[0]/aspect);lo=center-wh/2;hi=center+wh/2
            return [max(0,int(np.floor(lo[0]))),max(0,int(np.floor(lo[1]))),min(W,int(np.ceil(hi[0]))),min(H,int(np.ceil(hi[1])))]
        crops={'balls':roi([259,261,263],pad=8,aspect=2.8),'cups':roi([266,267],pad=8,aspect=2.8),'objects':roi([259,261,263,266,267],pad=32,aspect=1.5)}
        meta={'source':source,'checkpoint':str(a.checkpoint),'segmentation':str(a.segmentation),'instance_count':len(ins['instances']),'named_entities':len(set(filter(None,names.values()))),'segmentation_checkpoint_max_distance':float(dd.max()),'named_component_count':sum(bool(v) for v in names.values()),'palette':palette,'objects':objs,'crops':crops,'blend_weights':{'real':.5,'render':.5},'image_adjustments':'Calibration undistortion and shared crops only; no geometric post-registration, sharpening, denoising, exposure or color adjustment.'}
        (folder/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        print(source['key'],'crops',crops,flush=True)

if __name__=='__main__':main()
