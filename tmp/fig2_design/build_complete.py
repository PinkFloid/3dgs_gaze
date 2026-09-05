"""Publication figure: native PDF/SVG labels with real 3DGS render assets."""
from pathlib import Path
from io import BytesIO
import json, math, base64, html, os
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'output/fig2_complete'; A=OUT/'assets'
PDF=ROOT/'output/pdf/gazesplat_fig2_v2_complete.pdf'
PDF.parent.mkdir(exist_ok=True,parents=True)
W,H=2000,1120; SCALE=516/W
font_sets=[
    (('arial.ttf','arialbd.ttf','ariali.ttf'), [Path('C:/Windows/Fonts'),Path('/Library/Fonts')]),
    (('LiberationSans-Regular.ttf','LiberationSans-Bold.ttf','LiberationSans-Italic.ttf'),
     [Path('/usr/share/fonts/truetype/liberation2'),Path('/usr/share/fonts/truetype/liberation')]),
    (('DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSans-Oblique.ttf'),[Path('/usr/share/fonts/truetype/dejavu')]),
]
font_paths=None
for filenames, directories in font_sets:
    candidates=([Path(os.environ['GAZESPLAT_FONT_DIR'])] if os.environ.get('GAZESPLAT_FONT_DIR') else [])+directories
    for directory in candidates:
        paths=[directory/name for name in filenames]
        if all(p.is_file() for p in paths):
            font_paths=paths
            break
    if font_paths:break
if font_paths is None:
    raise RuntimeError('Install Arial, Liberation Sans, or DejaVu Sans, or set GAZESPLAT_FONT_DIR to their font directory.')
for key,font_path in zip(('R','B','I'),font_paths):
    pdfmetrics.registerFont(TTFont(key,str(font_path)))
c=canvas.Canvas(str(PDF),pagesize=(W*SCALE,H*SCALE),pageCompression=1)
c.setTitle('GazeSplat - 3DGS-mediated instance disambiguation')
c.setAuthor('GazeSplat')
c.scale(SCALE,SCALE)
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
INK='#253943'; MUT='#657681'; TEAL='#00877A'; PALE='#EAF5F2'; AMBER='#C28A2D'; AP='#F8EEDC'; LINE='#CFDADD'; LIGHT='#F4F7F8'; PURP='#8C8BB8'; BROWN='#BB9278'; WHITE='#FFFFFF'

def style(fill=None,stroke=None,lw=2):
    if fill:c.setFillColor(HexColor(fill))
    if stroke:c.setStrokeColor(HexColor(stroke))
    c.setLineWidth(lw)
def rect(x,y,w,h,fill=None,stroke=None,lw=2,r=0):
    style(fill,stroke,lw)
    if r:c.roundRect(x,H-y-h,w,h,r,fill=bool(fill),stroke=bool(stroke))
    else:c.rect(x,H-y-h,w,h,fill=bool(fill),stroke=bool(stroke))
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{lw}"/>')
def text(x,y,s,size=28,color=INK,font='R',anchor='start'):
    style(color);c.setFont(font,size)
    f={'start':c.drawString,'middle':c.drawCentredString,'end':c.drawRightString}[anchor];f(x,H-y,s)
    svg.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if font=="B" else 400}" font-style="{"italic" if font=="I" else "normal"}" fill="{color}" text-anchor="{anchor}">{html.escape(str(s))}</text>')
def line(x1,y1,x2,y2,color=LINE,lw=2,dash=None):
    style(stroke=color,lw=lw);c.setDash(dash or []);c.line(x1,H-y1,x2,H-y2);c.setDash([])
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{lw}"'+(f' stroke-dasharray="{",".join(map(str,dash))}"' if dash else '')+'/>')
def path(pts,fill=None,stroke=None,lw=2,close=False):
    p=c.beginPath();d=[]
    for cmd,*v in pts:
        d.append(cmd+' '+' '.join(map(str,v)))
        if cmd=='M':p.moveTo(v[0],H-v[1])
        elif cmd=='L':p.lineTo(v[0],H-v[1])
        elif cmd=='C':p.curveTo(v[0],H-v[1],v[2],H-v[3],v[4],H-v[5])
    if close:p.close();d.append('Z')
    style(fill,stroke,lw);c.drawPath(p,fill=bool(fill),stroke=bool(stroke))
    svg.append(f'<path d="{" ".join(d)}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{lw}" stroke-linecap="round" stroke-linejoin="round"/>')
def ellipse(x,y,rx,ry,fill=None,stroke=None,lw=2):
    style(fill,stroke,lw);c.ellipse(x-rx,H-y-ry,x+rx,H-y+ry,fill=bool(fill),stroke=bool(stroke))
    svg.append(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{lw}"/>')
def arrow(x1,y1,x2,y2,color=INK,lw=2.8,head=10,dash=None):
    line(x1,y1,x2,y2,color,lw,dash)
    ang=math.atan2(y2-y1,x2-x1);u=(math.cos(ang),math.sin(ang));v=(-u[1],u[0])
    path([('M',x2,y2),('L',x2-head*u[0]+head*.48*v[0],y2-head*u[1]+head*.48*v[1]),('L',x2-head*u[0]-head*.48*v[0],y2-head*u[1]-head*.48*v[1])],fill=color,close=True)
def badge(x,y,w,s,fill=PALE,color=TEAL,size=26):
    rect(x,y,w,42,fill,r=7);text(x+w/2,y+30,s,size,color,'B','middle')
def picture(p,x,y,w,h,crop=None):
    im=Image.open(p).convert('RGB')
    if crop:im=im.crop(crop)
    b=BytesIO();im.save(b,format='PNG');raw=b.getvalue()
    c.drawImage(ImageReader(BytesIO(raw)),x,H-y-h,w,h,mask='auto')
    uri='data:image/png;base64,'+base64.b64encode(raw).decode()
    svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" xlink:href="{uri}"/>')
def cross(x,y,r=10,color=AMBER,lw=2):
    line(x-r,y,x+r,y,color,lw);line(x,y-r,x,y+r,color,lw)
def subtext(x,y,base,sub,size=29,color=TEAL):
    text(x,y,base,size,color,'I');bw=pdfmetrics.stringWidth(base,'I',size);text(x+bw+1,y+6,sub,size*.65,color,'I')
    return x+bw+pdfmetrics.stringWidth(sub,'I',size*.65)+4

rect(0,0,W,H,WHITE)
# Open panels: white space and rules, without presentation-style container boxes.
text(36,46,'(a) Gaze + speech',33,INK,'B')
text(446,46,'(b) 3DGS-mediated instance disambiguation',33,TEAL,'B')
text(1630,46,'(c) Robot execution',33,INK,'B')
line(36,66,390,66);line(446,66,1570,66,TEAL,2.5);line(1630,66,1964,66)
text(36,102,'Synchronized user input',26,MUT)
text(446,104,'Offline',24,MUT,'B')
text(539,104,'Multiview RGB  →  metric 3DGS  →  persistent IDs',26,MUT)
text(1630,102,'Shared map coordinates',26,MUT)

# Wearer: a restrained vector drawing, not a fabricated eye-camera image.
path([('M',88,340),('C',94,313,116,303,151,294),('L',151,264),('C',125,249,115,213,120,183),('C',128,142,168,135,201,152),('C',222,163,228,183,226,203),('L',248,222),('L',229,229),('L',225,253),('C',220,269,202,276,185,274),('L',185,294),('C',220,304,240,317,247,340)],fill='#E7EEF0',stroke=INK,lw=3,close=True)
path([('M',123,187),('C',125,151,153,139,181,145),('C',207,146,222,165,226,190),('C',215,176,202,181,188,172),('C',167,181,141,182,123,187)],fill=INK,close=True)
ellipse(158,220,8,13,stroke=MUT,lw=2)
line(165,203,231,203,INK,5);rect(202,192,36,27,WHITE,INK,3,r=5);line(198,199,176,200,INK,3)
ellipse(231,205,5,5,AMBER);ellipse(243,204,4,4,TEAL)
for rr in (15,25):
    path([('M',250+rr*.25,204-rr*.55),('C',260+rr*.4,204-rr*.3,260+rr*.4,204+rr*.3,250+rr*.25,204+rr*.55)],stroke=AMBER,lw=2)
text(212,377,'Eye-tracker pose + gaze',26,INK,'B','middle')
text(212,413,'World-space fixation events',25,MUT,anchor='middle')
line(59,467,323,467,LINE,2)
for xx,yy,rr in [(76,462,4),(96,472,4),(135,461,5),(149,468,6),(168,463,7),(187,471,6),(204,465,5),(226,469,4),(276,473,4),(300,462,4)]:ellipse(xx,yy,rr,rr,AMBER)
rect(125,444,114,48,None,AMBER,2,r=7)
text(181,526,'fixation interval',25,AMBER,anchor='middle')
arrow(315,258,440,258,AMBER,3)
text(377,238,'pose + gaze',24,AMBER,anchor='middle')

# Actual model queried at wearer pose. A separate view, explicitly a rendering.
text(450,186,'Render at wearer pose',27,INK,'B')
picture(A/'query_rgb.png',450,209,290,174)
rect(450,209,290,174,None,LINE,1.6)
ellipse(595,296,30,30,None,AMBER,2.4);ellipse(595,296,16,16,None,AMBER,1.5);cross(595,296,8)
text(450,423,'Depth + opacity',27,INK,'B')
text(450,459,'View-dependent surfaces',25,MUT)

# Dominant 3DGS reconstruction, rendered directly from the trained checkpoint.
text(1183,158,'Persistent metric 3DGS map',31,TEAL,'B','middle')
crop=(245,148,1270,720); ix,iy,iw,ih=755,180,805,449.23
picture(A/'hero_1.png',ix,iy,iw,ih,crop)
arrow(793,287,750,287,TEAL,2.4)
metadata=json.loads((A/'render_metadata.json').read_text(encoding='utf-8'))
def uv(name):
    u,v=metadata['hero']['1']['objects'][name]['uv'];return ix+(u-crop[0])*iw/(crop[2]-crop[0]),iy+(v-crop[1])*ih/(crop[3]-crop[1])
for name,lab,xx,yy,col in [('ball_R','ball_R',1425,279,TEAL),('ball_M','ball_M',1299,447,MUT),('ball_L','ball_L',1043,492,MUT)]:
    ux,uy=uv(name)
    line(ux,uy+12,xx,yy-22,col,1.8)
    rect(xx-52,yy-32,105,40,WHITE,r=5)
    text(xx,yy,lab,27,col,'B','middle')
text(450,562,'Uncertain gaze',27,AMBER,'B')
text(450,599,'Query a local angular patch',25,MUT)
arrow(594,477,594,522,AMBER,2.4)
arrow(594,614,594,665,AMBER,2.4)
line(450,642,1570,642)

# Voice stream, independent of map query.
line(36,570,389,570)
text(36,614,'Speech with word times',27,INK,'B')
wave=[4,7,4,9,17,26,13,31,37,22,11,18,7,10,22,35,20,28,42,29,13,9,17,7,4,9,16,28,12,8,5]
for n,hh in enumerate(wave):line(48+n*10.5,666-hh*.5,48+n*10.5,666+hh*.5,AMBER,3)
text(36,731,'“Bring me that ball.”',32,INK,'I')
line(220,743,278,743,AMBER,3)
arrow(212,766,212,796,INK,2.4)
rect(36,808,354,87,LIGHT,r=8)
text(56,843,'LLM: actions + slots',28,INK,'B')
text(56,879,'object = “that ball”',26,MUT)
arrow(212,907,212,957,INK,2.4)

# The actual local query is a controlled, illustrative ray in the real model.
text(450,686,'1  Visible surface evidence',27,INK,'B')
for name,xx,cap in [('query_depth.png',450,'Depth'),('query_alpha.png',577,'Opacity'),('query_instances.png',704,'Instances')]:
    picture(A/name,xx,711,108,108);rect(xx,711,108,108,None,LINE,1.2);text(xx+54,847,cap,24,MUT,anchor='middle')
text(631,775,'α ≈ 1',26,MUT,anchor='middle')
for rr in (18,35,49):ellipse(758,765,rr,rr,None,AMBER,1.4)
cross(758,765,7,AMBER,1.7)
prefix='Weighted surface mass '
text(450,891,prefix,24,MUT)
subtext(450+pdfmetrics.stringWidth(prefix,'R',24),891,'q','k',24,MUT)
arrow(828,767,867,767,INK,2.3)

text(895,686,'2  Size-aware capture',27,INK,'B')
ellipse(947,768,49,49,AP)
for rr in (21,34,48):ellipse(947,768,rr,rr,None,'#D9C6A1',1.4)
ellipse(947,768,20,20,TEAL)
text(1026,751,'Expected',25,MUT);text(1026,783,'capture C',25,MUT);text(1133,789,'k',18,MUT,'I')
text(895,850,'capture',31,TEAL,'I');text(995,857,'k',21,TEAL,'I');text(1014,850,'= q',31,TEAL,'I');text(1063,857,'k',21,TEAL,'I');text(1086,850,'/ C',31,TEAL,'I');text(1130,857,'k',21,TEAL,'I')
text(895,891,'Normalize by target angular size',24,MUT)
arrow(1190,767,1227,767,INK,2.3)

text(1253,686,'3  Rank + binding gate',27,INK,'B')
text(1253,717,'Illustrative map query',23,MUT,'I')
rank=metadata['query']['rank']; cand=rank['candidates']
for j,(label,col) in enumerate([('ball_R',TEAL),('cup_1',PURP),('red cup',BROWN)]):
    yy=754+j*38;text(1253,yy+6,label,24,col,'B' if j==0 else 'R')
    rect(1360,yy-13,136,17,LIGHT,r=3)
    val=cand[j]['capture'];rect(1360,yy-13,136*val/1.3,17,col,r=3);text(1554,yy+4,f'{val:.2f}',24,col,anchor='end')
badge(1253,866,138,'ball_R');text(1409,895,'or no binding',25,MUT)

# Recorded robot photographs are separate from the illustrative query.
picture(ROOT/'paper/fig2_assets/robot_nav.jpg',1630,131,334,314,(772,114,1372,678))
text(1630,482,'Navigate & re-observe',28,INK,'B')
arrow(1797,499,1797,531,INK,2.4)
picture(ROOT/'paper/fig2_assets/delivery.jpg',1630,546,334,314,(1060,440,1740,1080))
text(1630,899,'Grasp & deliver',28,INK,'B')

# Final instruction grounding lane. No confirmation branch in voice mode.
line(36,942,1964,942,LINE,2)
text(36,989,'(d) Temporal grounding',29,INK,'B')
text(36,1030,'Actions + referring expressions',24,MUT)
arrow(389,1024,439,1024,INK,2.5)
text(464,990,'gaze',25,MUT);line(535,982,849,982,LINE,2)
rect(636,963,151,36,PALE,TEAL,1.4,r=3);text(711,990,'ball_R',25,TEAL,'B','middle')
text(464,1043,'word',25,MUT);line(535,1035,849,1035,LINE,2)
line(707,997,707,1041,AMBER,1.7,[4,3]);ellipse(707,1035,5,5,AMBER)
text(707,1075,'“that”',25,AMBER,'B','middle')
arrow(870,1024,920,1024,INK,2.5)
badge(941,1001,228,'that → ball_R',PALE,TEAL,27)
arrow(1186,1024,1230,1024,INK,2.5)
text(1253,990,'Executable commands',28,INK,'B')
text(1253,1030,'action + ID + world hint',25,MUT)
text(1253,1074,'Dispatch when grounded',25,TEAL,'B')
line(1545,1024,1602,1024,TEAL,2.5)
line(1602,1024,1602,291,TEAL,2.5)
arrow(1602,291,1624,291,TEAL,2.5)
# Link successful object resolution to the time-aligned instance stream.
line(1322,913,1322,929,TEAL,1.8)
line(711,929,1322,929,TEAL,1.8)
arrow(711,929,711,957,TEAL,1.8,8)

c.showPage();c.save();svg.append('</svg>')
(OUT/'gazesplat_fig2_v2_complete.svg').write_text('\n'.join(svg),encoding='utf-8')
print(PDF)
print(OUT/'gazesplat_fig2_v2_complete.svg')
