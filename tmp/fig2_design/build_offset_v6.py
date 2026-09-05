"""Readable full-origin example using unchanged recorded low gaze."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
NEW=ROOT/'output/fig2_offset_v6/assets'
for name in ['hero.png','grasp_recorded.png','provenance.json','grasp_provenance.json']:
    if not (NEW/name).is_file():
        raise FileNotFoundError(f'Missing committed figure asset: {NEW/name}')
src=Path(__file__).with_name('build_recorded_v3.py').read_text(encoding='utf-8')
src=src.replace('output/fig2_recorded_v3','output/fig2_offset_v6').replace('gazesplat_fig2_recorded_v3','gazesplat_fig2_offset_v6')
start=src.index('# The same origin and gaze direction');end=src.index('cards=[',start)
block=r'''
geom=json.loads((A/'geometry.json').read_text());view=geom['views']['3'];proj=view['projection']
text(758,235,'3DGS-based instance disambiguation',31.5,TEAL,'B','middle')
text(758,274,'A low gaze still supports the target',30.2,MUT,anchor='middle')
gcrop=(40,320,1460,805);gx,gy,gw=449,306,616;scale=gw/(gcrop[2]-gcrop[0]);gh=scale*(gcrop[3]-gcrop[1])
picture(A/'full_3.png',gx,gy,gw,gh,gcrop)
def uv(p):return gx+(p[0]-gcrop[0])*scale,gy+(p[1]-gcrop[1])*scale
ox,oy=uv(proj['origin']);cx,cy=uv(proj['central_surface_point'])
# The center of this source icon is exactly the projected recorded origin.
ellipse(ox,oy,15,10,WHITE,AMBER,2.)
ellipse(ox,oy,4.2,4.2,AMBER)
text(ox-18,oy-24,'Eye origin',30.2,AMBER,'B')
text(899,321,'±2σ query',30.2,AMBER,'I','middle')
arrow(ox+.39*(cx-ox),oy+.39*(cy-oy),ox+.59*(cx-ox),oy+.59*(cy-oy),AMBER,2.2,9)
cross(cx,cy,7.5,WHITE,4.8);cross(cx,cy,7.5,AMBER,2.5)
text(467,456,'Low gaze',30.2,AMBER,'I')
line(600,450,668,450,AMBER,1.3);line(668,450,cx-9,cy+7,AMBER,1.3)
for name,dx in [('ball_L',-28),('ball_M',0),('ball_R',30)]:
    pts=[uv(v) for v in proj['boxes'][name]];px=sum(v[0] for v in pts)/8;py=max(v[1] for v in pts)
    tx=px+dx;col=TEAL if name=='ball_M' else MUT
    line(px,py+3,tx,473,col,1.2)
    rect(tx-47,478,94,33,WHITE,r=3)
    text(tx,503,name,30.2,col,'B','middle')
text(758,536,'Weighted evidence among objects',30.2,MUT,anchor='middle')
shares=geom['query']['object_shares'];ss=[shares['球M'],shares['球L'],1-shares['球M']-shares['球L']]
for xx,ww,label,share,color in [(449,185,'ball_M',ss[0],TEAL),(654,193,'ball_L',ss[1],'#75899A'),(869,197,'others',ss[2],'#95A4AF')]:
    rect(xx,547,ww,54,'#EDF7F3' if label=='ball_M' else LIGHT,r=5)
    text(xx+ww/2,573,f'{label}  {share:.0%}',30.2,color,'B','middle')
    rect(xx+12,582,ww-24,8,'#DFE6E9',r=3)
    rect(xx+12,582,(ww-24)*share,8,color,r=3)

'''
src=src[:start]+block+src[end:]
exec(compile(src,'offset_figure_v6','exec'))
