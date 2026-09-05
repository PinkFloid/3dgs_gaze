"""Refine the front spatial illustration and crop the execution photo leftward."""
from pathlib import Path

flow_source=Path(__file__).with_name('build_flow_v2.py').read_text(encoding='utf-8')
flow_source=flow_source.replace('output/fig2_flow_v2','output/fig2_refined_v2')
flow_source=flow_source.replace('gazesplat_fig2_v2_flow','gazesplat_fig2_v2_refined')
flow_source=flow_source.replace(
    "picture(ROOT/'paper/fig2_assets/delivery.jpg',1804,412,342,363.38,(1060,400,1700,1080))",
    "picture(ROOT/'paper/fig2_assets/delivery.jpg',1832.5,402,285,384.42,(1050,500,1480,1080))"
)

wearer_start=flow_source.index('# Wearer gaze -> persistent world fixation events.')
wearer_end=flow_source.index('# v2: retain the original fan/cone intuition')
wearer_block=r'''
# A light vertical sequence replaces the stack of large text-filled cards.
text(54,235,'Wearer gaze',32,AMBER,'B')
text(54,274,'→ world fixations',32,AMBER,'B')
for cy in (337,447,557):ellipse(83,cy,23,23,AP)
arrow(83,369,83,411,AMBER,1.8,7)
arrow(83,479,83,521,AMBER,1.8,7)
# Localization camera symbol.
rect(68,328,24,18,None,AMBER,2,r=3)
path([('M',92,331),('L',101,326),('L',101,348),('L',92,343)],fill=AP,stroke=AMBER,lw=2,close=True)
ellipse(80,337,4,4,None,AMBER,1.6)
# A gaze ray expressed in a coordinate frame.
arrow(74,455,99,442,AMBER,1.8,5)
line(74,455,74,431,AMBER,1.6);line(74,455,96,457,AMBER,1.6)
ellipse(74,455,2.5,2.5,AMBER)
for xx,yy in [(75,550),(83,555),(91,550),(78,561),(89,562)]:ellipse(xx,yy,2.7,2.7,AMBER)
ellipse(84,556,15,12,None,AMBER,1.5)
for yy,title,detail in [(331,'Localize camera','ArUco map frame'),(441,'Lift gaze to 3D','Pose + surface depth'),(551,'Form fixation events','Cluster in space + time')]:
    text(121,yy,title,27,INK,'B')
    text(121,yy+39,detail,24,MUT)
line(72,648,345,648,LINE,1.7)
for xx,dy,rr in [(82,2,3.5),(103,-4,3.5),(162,-2,5),(176,4,5),(195,-3,6),(215,2,5),(233,-3,5),(282,3,3.5),(325,-2,3.5)]:ellipse(xx,648+dy,rr,rr,AMBER)
rect(149,630,99,38,None,AMBER,1.7,r=5)
text(207,701,'World-space fixation events',25,MUT,anchor='middle')
arrow(393,578,422,578,AMBER,2.8,9)

'''
flow_source=flow_source[:wearer_start]+wearer_block+flow_source[wearer_end:]

query_start=flow_source.index('# v2: retain the original fan/cone intuition')
query_end=flow_source.index('chips=[(450,181')
query_block=r'''
# Show the actual map-render query rather than an abstract funnel illustration.
text(758,235,'3DGS-based instance disambiguation',31,TEAL,'B','middle')
text(758,270,'Visible surfaces within gaze uncertainty',27,MUT,anchor='middle')
text(602,315,'Map view at wearer pose',26,INK,'B','middle')
text(950,315,'Local query',26,INK,'B','middle')
picture(A/'query_rgb.png',450,336,304,182.4)
rect(450,336,304,182.4,None,LINE,1.5)
# Patch corners correspond to a +/- 2 degree support in a 22 degree RGB view.
px,py,pr=602,427.2,27.4
for sx,sy in [(-1,-1),(1,-1),(-1,1),(1,1)]:
    ex,ey=px+sx*pr,py+sy*pr
    line(ex,ey,ex-sx*10,ey,AMBER,2.8)
    line(ex,ey,ex,ey-sy*10,AMBER,2.8)
ellipse(px,py,13.7,13.7,None,AMBER,1.8)
cross(px,py,6,AMBER,2)
arrow(774,427,838,427,AMBER,2.6,9)
picture(A/'query_instances.png',858,336,182,182)
rect(858,336,182,182,None,LINE,1.5)
for rr in (45.5,89):ellipse(949,427,rr,rr,None,AMBER,1.6)
cross(949,427,8,AMBER,1.8)
text(602,551,'Gaze + angular uncertainty',25,AMBER,anchor='middle')
text(949,551,'Depth + opacity + ID',25,TEAL,anchor='middle')

'''
flow_source=flow_source[:query_start]+query_block+flow_source[query_end:]
flow_source=flow_source.replace('rect(x,553,w,96,LIGHT,r=6);text(x+w/2,585,label,27',
    'rect(x,576,w,87,LIGHT,r=6);text(x+w/2,607,label,27')
flow_source=flow_source.replace(',628,',',646,')
flow_source=flow_source.replace('635,602,649,602','635,621,649,621').replace('854,602,870,602','854,621,870,621')

exec(compile(flow_source,'refined_figure','exec'))
