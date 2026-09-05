"""Fig. 2 revision with a recorded fixation, matched map, and clear grasp frame.

CPU-only layout builder. GPU rendering and raw recording extraction are separate.
"""
from pathlib import Path

# Reuse the established vector primitives only; layout is explicit below.
source=Path(__file__).with_name('build_complete.py').read_text(encoding='utf-8')
prefix=source.split('rect(0,0,W,H,WHITE)')[0]
prefix=prefix.replace("OUT=ROOT/'output/fig2_complete'; A=OUT/'assets'",
                      "OUT=ROOT/'output/fig2_recorded_v3'; A=OUT/'assets'")
prefix=prefix.replace('gazesplat_fig2_v2_complete.pdf','gazesplat_fig2_recorded_v3.pdf')
prefix=prefix.replace('W,H=2000,1120; SCALE=516/W','W,H=2200,1030; SCALE=(180/25.4*72)/W')
exec(compile(prefix,'figure_primitives','exec'))

BLUE='#406184'; BLUEP='#F3F6FA'; AMBERP='#FFFCF5'; GREENP='#F5FAF7'
meta=json.loads((A/'provenance.json').read_text())

rect(0,0,W,H,WHITE)
rect(30,24,2140,132,WHITE,LINE,2,r=12)
text(54,62,'Offline, once: metric mapping + persistent instance registration',33,INK,'B')
nodes=[(54,236,'Multiview RGB'),(324,375,'COLMAP + metric frame'),
       (733,305,'Metric 3DGS training'),(1072,461,'SAM + cross-view association'),
       (1567,579,'Persistent IDs + one-time naming')]
for x,w,s in nodes:
    rect(x,86,w,46,LIGHT,r=6); text(x+w/2,118,s,30.2,INK,anchor='middle')
for a,b in zip(nodes[:-1],nodes[1:]): arrow(a[0]+a[1]+6,109,b[0]-7,109,INK,2.6,9)
arrow(1435,160,1435,186,TEAL,2.6,9)

rect(30,190,360,540,AMBERP,'#DECFAE',2,r=12)
rect(425,190,665,540,WHITE,'#B7CFC9',2.5,r=12)
rect(1125,190,620,540,WHITE,'#B7CFC9',2.5,r=12)
rect(1780,190,390,805,BLUEP,'#C3D0DF',2,r=12)
rect(30,770,1715,225,GREENP,'#C6D9D0',2,r=12)

# Recorded input is transformed into a world-space fixation.
text(54,235,'Wearer gaze',33,AMBER,'B')
text(54,274,'→ world fixations',33,AMBER,'B')
for cy in (337,447,557): ellipse(80,cy,22,22,AP)
arrow(80,369,80,411,AMBER,1.8,7); arrow(80,479,80,521,AMBER,1.8,7)
rect(66,328,24,18,None,AMBER,2,r=3)
path([('M',90,331),('L',99,326),('L',99,348),('L',90,343)],fill=AP,stroke=AMBER,lw=2,close=True)
ellipse(78,337,4,4,None,AMBER,1.6)
arrow(71,455,96,442,AMBER,1.8,5)
line(71,455,71,431,AMBER,1.6); line(71,455,93,457,AMBER,1.6)
ellipse(71,455,2.5,2.5,AMBER)
for xx,yy in [(72,550),(80,555),(88,550),(75,561),(86,562)]: ellipse(xx,yy,2.7,2.7,AMBER)
ellipse(81,556,15,12,None,AMBER,1.5)
for y,title,detail in [(331,'Localize camera','Surveyed ArUco'),(441,'Lift gaze to 3D','Pose + depth'),(551,'Form fixations','Space + time')]:
    text(116,y,title,31,INK,'B');text(116,y+39,detail,30.2,MUT)
line(70,648,347,648,LINE,1.7)
for xx,dy,rr in [(82,2,3.5),(103,-4,3.5),(162,-2,5),(176,4,5),(195,-3,6),(215,2,5),(233,-3,5),(282,3,3.5),(325,-2,3.5)]: ellipse(xx,648+dy,rr,rr,AMBER)
rect(149,630,99,38,None,AMBER,1.7,r=5)
text(210,703,'World-frame events',30.2,MUT,anchor='middle')
arrow(393,545,422,545,AMBER,2.8,9)

# The same origin and gaze direction determine the full render and square patch.
text(758,235,'3DGS-based instance disambiguation',31.5,TEAL,'B','middle')
text(758,274,'Visible surfaces within gaze uncertainty',30.2,MUT,anchor='middle')
text(641,320,'Fixation-view render',30.2,INK,'B','middle')
text(963,320,'Local query',30.2,INK,'B','middle')
vx,vy,vw,vh=449,336,384,384*526/920
view_crop=(420,100,1340,626)
picture(A/'fixation_view.png',vx,vy,vw,vh,view_crop)
rect(vx,vy,vw,vh,None,LINE,1.5)
px=vx+(700-view_crop[0])*vw/(view_crop[2]-view_crop[0])
py=vy+(400-view_crop[1])*vh/(view_crop[3]-view_crop[1])
f=meta['view']['K'][0][0]; pr=f*math.tan(math.radians(2))*vw/(view_crop[2]-view_crop[0])
rect(px-pr,py-pr,2*pr,2*pr,None,AMBER,2.6)
cross(px,py,7,WHITE,4.2);cross(px,py,7,AMBER,2.1)
# Connect the actual angular ROI to the enlarged query, avoiding an unrelated box.
qx,qy,qs=857,336,208
line(px+pr,py-pr,qx,qy,AMBER,1.5)
line(px+pr,py+pr,qx,qy+qs,AMBER,1.5)
picture(A/'query_instances.png',qx,qy,qs,qs)
rect(qx,qy,qs,qs,None,AMBER,1.8)
for r in [qs*.125,qs*.25]: ellipse(qx+qs/2,qy+qs/2,r,r,None,AMBER,1.6)
cross(qx+qs/2,qy+qs/2,8,WHITE,4.2);cross(qx+qs/2,qy+qs/2,8,AMBER,2.)
text(641,590,'Recorded fixation',30.2,AMBER,anchor='middle')
text(961,590,'33 × 33 · ±2σ',30.2,MUT,anchor='middle')

cards=[(449,185,'Visible mass'),(654,193,'Normalize'),(869,197,'Rank + gate')]
for x,w,t in cards: rect(x,610,w,74,LIGHT,r=6);text(x+w/2,639,t,30.2,INK,'B','middle')
xx=subtext(465,674,'q','k',31,TEAL);text(xx+3,674,'=',31,TEAL,'I')
xx=subtext(xx+25,674,'m','k',31,TEAL);text(xx+3,674,'/ W',31,TEAL,'I')
xx=subtext(670,674,'S','k',31,TEAL);text(xx+3,674,'=',31,TEAL,'I')
xx=subtext(xx+25,674,'q','k',31,TEAL);text(xx+1,674,'/',31,TEAL,'I');subtext(xx+16,674,'C','k',31,TEAL)
text(967,674,'ID / reject',30.2,TEAL,'B','middle')
arrow(636,654,650,654,INK,2.,6);arrow(850,654,865,654,INK,2.,6)
text(758,716,'Size + distance · candidate competition',30.2,MUT,anchor='middle')

# The original successful hero camera is retained, with the map matching the trial.
text(1435,235,'Shared 3DGS instance map',33,TEAL,'B','middle')
text(1435,276,'One metric frame for user and robot',30.2,MUT,anchor='middle')
crop=(245,148,1270,720);ix,iy,iw,ih=1145,301,580,323.67
picture(A/'hero.png',ix,iy,iw,ih,crop)
for name,x,y,col in [('ball_L',1352,530,MUT),('ball_M',1550,496,TEAL),('ball_R',1653,375,MUT)]:
    u,v=meta['hero']['objects'][name]['uv'];u=ix+(u-crop[0])*iw/(crop[2]-crop[0]);v=iy+(v-crop[1])*ih/(crop[3]-crop[1])
    line(u,v+5,x,y-13,col,1.6);rect(x-53,y-30,106,40,WHITE,r=4)
    text(x,y,name,30.2,col,'B','middle')
text(1435,661,'Depth + opacity + instance labels',30.2,TEAL,'B','middle')
text(1435,705,'Persistent IDs + metric locations',30.2,MUT,anchor='middle')
arrow(1122,454,1093,454,TEAL,2.8,9)
arrow(1748,342,1777,342,TEAL,2.8,9)

# Physical execution: the target and both gripper fingers are fully visible.
text(1804,235,'Robot execution',33,BLUE,'B')
text(1804,278,'Grounded target → action',30.2,MUT)
rect(1804,308,342,90,'#E6EDF5',r=7)
text(1975,342,'Navigate in map frame',30.2,BLUE,'B','middle')
text(1975,382,'Reobserve target',30.2,MUT,anchor='middle')
picture(A/'grasp_recorded.png',1804,420,342,378.38298)
text(1975,838,'Grasp + carry',32,BLUE,'B','middle')
line(1804,859,2146,859,'#C3D0DF',1.7)
rect(1804,884,342,90,'#E6EDF5',r=7)
text(1975,920,'Ordered commands',30.2,BLUE,'B','middle')
text(1975,957,'move · pick · place',30.2,MUT,anchor='middle')

# Commands are grounded by each referring word's timestamp.
text(54,815,'Speech → grounded commands',33,TEAL,'B')
for x,w in [(54,257),(351,352),(747,496),(1287,434)]: rect(x,847,w,132,'#E8F2ED',r=7)
text(182,890,'Streaming ASR',31,INK,'B','middle')
text(182,938,'Word timestamps',30.2,MUT,anchor='middle')
text(527,890,'LLM: actions + slots',31,INK,'B','middle')
text(527,938,'“Bring me this ball.”',30.2,MUT,'I','middle')
text(995,884,'Word–fixation binding',31,INK,'B','middle')
text(768,924,'gaze',30.2,MUT);line(847,918,1217,918,LINE,2)
rect(967,901,145,34,WHITE,TEAL,1.5,r=4);text(1039,927,'ball_M',30.2,TEAL,'B','middle')
text(768,960,'word',30.2,MUT);line(847,952,1217,952,LINE,2)
line(1028,936,1028,957,AMBER,1.8,[3,3]);ellipse(1028,952,4,4,AMBER)
rect(1060,939,99,34,'#E8F2ED');text(1066,964,'“this”',30.2,AMBER,'B')
text(1504,886,'Compile + dispatch',31,INK,'B','middle')
text(1504,925,'Action + ID + world arguments',30.2,MUT,anchor='middle')
text(1504,962,'On successful grounding',30.2,TEAL,'B','middle')
for xa,xb in [(315,347),(707,743),(1247,1283)]: arrow(xa,911,xb,911,INK,2.6,9)
arrow(1748,911,1777,911,TEAL,2.8,9)
line(758,733,758,752,TEAL,2.3);line(758,752,995,752,TEAL,2.3)
arrow(995,752,995,844,TEAL,2.3,9)

c.showPage();c.save();svg.append('</svg>')
(OUT/'gazesplat_fig2_recorded_v3.svg').write_text('\n'.join(svg),encoding='utf-8')
print(PDF)
