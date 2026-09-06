"""Compose the measured camera comparison as PDF and editable SVG.

Only shared rectangular crops and vector annotations are added to the source
assets. The images are never registered, retouched, or color-matched here.
"""
from pathlib import Path
import argparse
import base64
import html
import json
import os

from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'output/scene_map_comparison_iphone')
    parser.add_argument('--pdf', type=Path, default=ROOT / 'output/pdf/gazesplat_scene_map_comparison_iphone.pdf')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out_dir / 'panels'
    crops_dir.mkdir(exist_ok=True)
    meta = json.loads((args.asset_dir / 'metadata.json').read_text(encoding='utf-8-sig'))
    if not meta['source']['source_type'].startswith('iPhone'):
        raise ValueError('Use iPhone construction-image assets; later recordings contain moved objects')
    fonts = {'Sans': 'Helvetica', 'SansBold': 'Helvetica-Bold'}
    font_dirs = [Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts',
                 Path('/usr/share/fonts/truetype/liberation'),
                 Path('/usr/share/fonts/truetype/liberation2')]
    for folder in font_dirs:
        for regular, bold in [('arial.ttf', 'arialbd.ttf'), ('LiberationSans-Regular.ttf', 'LiberationSans-Bold.ttf')]:
            if (folder/regular).exists() and (folder/bold).exists():
                for name, file in [('Sans', regular), ('SansBold', bold)]:
                    pdfmetrics.registerFont(TTFont(name, str(folder/file)))
                    fonts[name] = name
                break
        if fonts['Sans'] == 'Sans':
            break

    W, H = 510, 239
    margin, gap = 2, 6
    pw = (W - 2 * margin - 3 * gap) / 4
    xs = [margin + i * (pw + gap) for i in range(4)]
    pdf = canvas.Canvas(str(args.pdf), pagesize=(W, H), pageCompression=1)
    pdf.setTitle('GazeSplat: real scene and persistent-instance map')
    pdf.setAuthor('GazeSplat')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{180*H/W:.4f}mm" viewBox="0 0 {W} {H}">',
           '<title>Real scene, 3DGS rendering, equal-weight overlay, and persistent-instance map</title>',
           f'<desc>Real images: {html.escape(meta["source"]["source_type"])}. All columns use the same calibrated camera and crop. No post-render registration is applied.</desc>',
           f'<rect width="{W}" height="{H}" fill="white"/>']

    def text(x, y, value, size=8, bold=False, color='#17232E', anchor='start'):
        font = fonts['SansBold' if bold else 'Sans']
        width = pdfmetrics.stringWidth(value, font, size)
        left = x - (width / 2 if anchor == 'middle' else width if anchor == 'end' else 0)
        assert left >= -0.1 and left + width <= W + 0.1, (value, left, width)
        pdf.setFont(font, size)
        pdf.setFillColor(HexColor(color))
        pdf.drawString(left, H - y, value)
        svg.append(f'<text x="{x}" y="{y}" font-family="Liberation Sans, Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">{html.escape(value)}</text>')

    def rect(x, y, w, h, stroke=None, fill=None, sw=0.5):
        if stroke:
            pdf.setStrokeColor(HexColor(stroke))
        if fill:
            pdf.setFillColor(HexColor(fill))
        pdf.setLineWidth(sw)
        pdf.rect(x, H-y-h, w, h, stroke=int(bool(stroke)), fill=int(bool(fill)))
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" stroke="{stroke or "none"}" fill="{fill or "none"}" stroke-width="{sw}"/>')

    def panel(name, row, crop, x, y, height):
        im = Image.open(args.asset_dir / f'{name}.png').convert('RGB')
        if crop:
            im = im.crop(crop)
        path = crops_dir / f'{row}_{name}.png'
        im.save(path)
        pdf.drawImage(str(path), x, H-y-height, width=pw, height=height)
        data = base64.b64encode(path.read_bytes()).decode('ascii')
        svg.append(f'<image x="{x}" y="{y}" width="{pw}" height="{height}" preserveAspectRatio="none" href="data:image/png;base64,{data}"/>')
        rect(x, y, pw, height, stroke='#BCC3CA', sw=0.35)

    def legend(ids, y):
        widths = [pdfmetrics.stringWidth(str(i), fonts['Sans'], 7) + 15 for i in ids]
        x = xs[3] + (pw - sum(widths)) / 2
        for iid, width in zip(ids, widths):
            rect(x, y-5.5, 5.5, 5.5, fill=meta['palette'][str(iid)]['hex'])
            text(x+8, y, str(iid), size=7)
            x += width

    names = ['real', 'render', 'blend', 'instances']
    titles = ['(a) Real scene', '(b) 3DGS rendering', '(c) Real + 3DGS', '(d) Instance map']
    is_phone = meta['source']['source_type'].startswith('iPhone')
    subtitles = ['iPhone capture' if is_phone else 'Pupil Core capture', 'Same camera pose', '50% real + 50% render', 'Persistent instance colors']
    for x, title, sub in zip(xs, titles, subtitles):
        text(x+pw/2, 10, title, size=9, bold=True, anchor='middle')
        text(x+pw/2, 22, sub, size=7, color='#53616D', anchor='middle')

    overview_y = 28
    source_w, source_h = meta['source']['size']
    # Portrait iPhone captures use a common 16:9 window centered vertically on
    # the task region, without independent image shifts or geometric changes.
    overview_crop = [0, 0, source_w, source_h]
    if source_h/source_w > 9/16:
        crop_h = round(source_w*9/16)
        object_y = (meta['crops']['objects'][1]+meta['crops']['objects'][3])/2
        top = max(0, min(source_h-crop_h, round(object_y-crop_h/2)))
        overview_crop = [0, top, source_w, top+crop_h]
    overview_h = pw * (overview_crop[3]-overview_crop[1]) / (overview_crop[2]-overview_crop[0])
    for x, name in zip(xs, names):
        panel(name, 'overview', overview_crop, x, overview_y, overview_h)
        roi = meta['crops']['objects']
        scale = pw/source_w
        bx, by = x + roi[0]*scale, overview_y + (roi[1]-overview_crop[1])*scale
        bw, bh = (roi[2]-roi[0])*scale, (roi[3]-roi[1])*scale
        rect(bx, by, bw, bh, stroke='#17232E', sw=1.3)
        rect(bx, by, bw, bh, stroke='#FFFFFF', sw=0.65)

    for row, label, y, ids in [
        ('balls', 'Three identical tennis balls', 115, [259, 261, 263]),
        ('cups', 'Two identical white cups', 184, [266, 267]),
    ]:
        text(margin, y, label, size=8.3, bold=True)
        legend(ids, y)
        crop = meta['crops'][row]
        height = pw * (crop[3]-crop[1])/(crop[2]-crop[0])
        assert y+7+height <= H-2
        for x, name in zip(xs, names):
            panel(name, row, crop, x, y+7, height)

    pdf.showPage()
    pdf.save()
    svg.append('</svg>')
    (args.out_dir / 'gazesplat_scene_map_comparison_iphone.svg').write_text('\n'.join(svg), encoding='utf-8')
    provenance = {
        'source_asset_dir': str(args.asset_dir),
        'source_metadata': str(args.asset_dir / 'metadata.json'),
        'page_points': [W, H],
        'panel_order': names,
        'overview_crop': overview_crop,
        'detail_crops': {k:meta['crops'][k] for k in ['balls','cups']},
        'overview_box': meta['crops']['objects'],
        'note': 'The overview rectangle marks the joint task-object region, not an object detection box. Every image in each row uses the same crop and display scale. Colored legend numbers are persistent instance IDs.',
    }
    (args.out_dir / 'figure_manifest.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(args.pdf)


if __name__ == '__main__':
    main()
