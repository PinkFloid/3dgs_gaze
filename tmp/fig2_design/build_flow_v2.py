"""Restore Fig. 2 v7's flow architecture, retaining current v2 semantics and renders."""
from pathlib import Path

# Reuse the native PDF/SVG primitives from the completed visual treatment.
src=Path(__file__).with_name('build_complete.py').read_text(encoding='utf-8')
prefix=src.split('rect(0,0,W,H,WHITE)')[0]
prefix=prefix.replace("OUT=ROOT/'output/fig2_complete'; A=OUT/'assets'", "OUT=ROOT/'output/fig2_flow_v2'; OUT.mkdir(parents=True,exist_ok=True); A=ROOT/'output/fig2_complete/assets'")
prefix=prefix.replace('gazesplat_fig2_v2_complete.pdf','gazesplat_fig2_v2_flow.pdf')
prefix=prefix.replace('W,H=2000,1120','W,H=2200,1030')
exec(compile(prefix,'figure_primitives','exec'))

BLUE='#406184'; BLUEP='#F3F6FA'; AMBERP='#FFFCF5'; GREENP='#F5FAF7'
rect(0,0,W,H,WHITE)

# The original overview structure: offline strip, two converging spatial streams,
# a separate speech lane, and a robot module receiving map and command inputs.
rect(30,24,2140,132,WHITE,LINE,2,r=12)
text(54,62,'Offline, once: metric mapping + instance registration',31,INK,'B')
nodes=[(54,236,'Multiview capture'),(324,375,'COLMAP + metric alignment'),(733,305,'3D Gaussian Splatting'),(1072,461,'SAM + cross-view association'),(1567,579,'Persistent IDs + one-time naming')]
for x,w,s in nodes:
    rect(x,86,w,46,LIGHT,r=6);text(x+w/2,118,s,27,INK,anchor='middle')
for a,b in zip(nodes[:-1],nodes[1:]):arrow(a[0]+a[1]+6,109,b[0]-7,109,INK,2.6,9)
arrow(1435,160,1435,186,TEAL,2.6,9)

rect(30,190,360,540,AMBERP,'#DECFAE',2,r=12)
rect(425,190,665,540,WHITE,'#B7CFC9',2.5,r=12)
rect(1125,190,620,540,WHITE,'#B7CFC9',2.5,r=12)
rect(1780,190,390,805,BLUEP,'#C3D0DF',2,r=12)
rect(30,770,1715,225,GREENP,'#C6D9D0',2,r=12)

# Wearer gaze -> persistent world fixation events.
text(54,235,'Wearer gaze',32,AMBER,'B')
text(54,274,'→ world fixations',32,AMBER,'B')
for y,title,detail in [(312,'Camera localization','ArUco landmarks'),(426,'World-space gaze','Pose + rendered depth'),(540,'Fixation events','Spatial + temporal clustering')]:
    rect(54,y,312,76,AP,r=7)
    text(210,y+31,title,28,INK,'B','middle')
    text(210,y+62,detail,24,MUT,anchor='middle')
arrow(210,396,210,417,AMBER,2.3,8)
arrow(210,502,210,530,AMBER,2.3,8)
line(73,672,347,672,LINE,2)
for x,dy,r in [(84,3,4),(111,-3,4),(158,0,6),(175,-4,6),(191,4,6),(209,0,6),(227,3,6),(244,-2,6),(294,3,4),(328,-3,4)]:ellipse(x,672+dy,r,r,AMBER)
rect(145,649,113,47,None,AMBER,1.8,r=5)
arrow(393,578,422,578,AMBER,2.8,9)

# v2: retain the original fan/cone intuition but show the actual square patch.
text(758,235,'3DGS-based instance disambiguation',31,TEAL,'B','middle')
text(758,270,'Visible surfaces within gaze uncertainty',27,MUT,anchor='middle')
path([('M',467,397),('L',835,308),('L',835,486)],fill=AP,close=True)
line(467,397,835,308,AMBER,2)
line(467,397,835,486,AMBER,2)
line(467,397,835,351,'#DAC398',1.2)
line(467,397,835,445,'#DAC398',1.2)
line(467,397,835,397,INK,2,[8,5])
ellipse(467,397,5,5,INK)
subtext(451,441,'F','k',28,INK)
path([('M',655,351),('C',655,375,696,376,696,397),('C',696,418,655,419,655,443),('L',655,351)],fill='#EAD2A5',stroke=AMBER,lw=1.7,close=True)
text(670,509,'Gaze kernel',26,AMBER,anchor='middle')
picture(A/'query_instances.png',835,308,178,178)
rect(835,308,178,178,None,LINE,1.7)
for rr in (43,84):ellipse(924,397,rr,rr,None,AMBER,1.8)
cross(924,397,10,AMBER,2)
text(924,521,'±2σ patch',26,MUT,anchor='middle')

chips=[(450,181,'Visible mass'),(653,197,'Size correction'),(874,191,'Rank + gate')]
for x,w,label in chips:
    rect(x,553,w,96,LIGHT,r=6);text(x+w/2,585,label,27,INK,'B','middle')
# Explicit v2 score, no posterior terminology or single-q threshold.
xx=subtext(478,628,'q','k',28,TEAL);text(xx+3,628,'=',27,TEAL,'I')
xx=subtext(xx+24,628,'m','k',28,TEAL);text(xx+3,628,'/ W',27,TEAL,'I')
xx=subtext(704,628,'q','k',29,TEAL);text(xx+5,628,'/',29,TEAL,'I');subtext(xx+22,628,'C','k',29,TEAL)
text(969,628,'ID / no binding',25,TEAL,'B','middle')
arrow(635,602,649,602,INK,2.2,7);arrow(854,602,870,602,INK,2.2,7)
text(758,694,'Visibility · uncertainty · target angular size',27,MUT,anchor='middle')

# Persistent map remains a peer of disambiguation, as in the original figure.
text(1435,235,'Shared 3DGS instance map',32,TEAL,'B','middle')
text(1435,276,'One metric frame for user and robot',27,MUT,anchor='middle')
crop=(245,148,1270,720); ix,iy,iw,ih=1145,301,580,323.67
picture(A/'hero_1.png',ix,iy,iw,ih,crop)
metadata=json.loads((A/'render_metadata.json').read_text(encoding='utf-8'))
def uv(name):
    u,v=metadata['hero']['1']['objects'][name]['uv'];return ix+(u-crop[0])*iw/(crop[2]-crop[0]),iy+(v-crop[1])*ih/(crop[3]-crop[1])
for name,lab,xx,yy,col in [('ball_R','ball_R',1628,383,TEAL),('ball_M','ball_M',1534,491,MUT),('ball_L','ball_L',1355,528,MUT)]:
    ux,uy=uv(name);line(ux,uy+5,xx,yy-12,col,1.5)
    rect(xx-49,yy-28,99,37,WHITE,r=4);text(xx,yy,lab,26,col,'B','middle')
text(1435,661,'Pose-conditioned depth + opacity',28,TEAL,'B','middle')
text(1435,705,'Persistent IDs + metric locations',27,MUT,anchor='middle')
arrow(1122,454,1093,454,TEAL,2.8,9)
arrow(1748,342,1777,342,TEAL,2.8,9)

# Robot module: map coordinates and grounded command are distinct inputs.
text(1804,235,'Robot execution',32,BLUE,'B')
text(1804,278,'Grounded target → action',27,MUT)
rect(1804,308,342,76,'#E6EDF5',r=7)
text(1975,340,'Map-frame navigation',28,BLUE,'B','middle')
text(1975,371,'Target re-observation',26,MUT,anchor='middle')
picture(ROOT/'paper/fig2_assets/delivery.jpg',1804,412,342,363.38,(1060,400,1700,1080))
text(1975,817,'Grasp & deliver',29,BLUE,'B','middle')
line(1804,841,2146,841,'#C3D0DF',1.7)
rect(1804,870,342,94,'#E6EDF5',r=7)
text(1975,908,'Execute commands',28,BLUE,'B','middle')
text(1975,944,'move · pick · place',27,MUT,anchor='middle')

# Speech processing remains one straight lower lane, with no confirmation stage.
text(54,815,'Speech → grounded commands',32,TEAL,'B')
for x,w in [(54,257),(351,352),(747,496),(1287,434)]:rect(x,847,w,132,'#E8F2ED',r=7)
text(182,890,'Streaming ASR',29,INK,'B','middle')
text(182,937,'Word timestamps',27,MUT,anchor='middle')
text(527,890,'LLM: actions + slots',29,INK,'B','middle')
text(527,937,'“Bring me that ball.”',27,MUT,'I','middle')
text(995,884,'Word–fixation binding',29,INK,'B','middle')
text(768,924,'gaze',25,MUT);line(839,918,1217,918,LINE,2)
rect(967,901,145,32,WHITE,TEAL,1.5,r=4);text(1039,925,'ball_R',26,TEAL,'B','middle')
text(768,959,'word',25,MUT);line(839,952,1217,952,LINE,2)
line(1028,934,1028,957,AMBER,1.8,[3,3]);ellipse(1028,952,4,4,AMBER)
rect(1060,939,94,34,'#E8F2ED');text(1066,963,'“that”',26,AMBER,'B')
text(1504,883,'Compile + dispatch',29,INK,'B','middle')
text(1504,922,'Action + ID + world parameters',26,MUT,anchor='middle')
text(1504,960,'On successful binding',25,TEAL,'B','middle')
for xa,xb in [(315,347),(707,743),(1247,1283)]:arrow(xa,911,xb,911,INK,2.6,9)
arrow(1748,911,1777,911,TEAL,2.8,9)
# Only the spatial resolution output joins the word/fixation binding stage.
line(758,733,758,752,TEAL,2.3)
line(758,752,995,752,TEAL,2.3)
arrow(995,752,995,844,TEAL,2.3,9)

c.showPage();c.save();svg.append('</svg>')
(OUT/'gazesplat_fig2_v2_flow.svg').write_text('\n'.join(svg),encoding='utf-8')
print(PDF)
