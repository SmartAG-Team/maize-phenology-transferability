"""Verify native raster dimensions and recorded artist geometry without vector exports."""
import json
from PIL import Image


def verify_png(folder, stem):
    layout = json.loads((folder / f'{stem}_layout.json').read_text())
    assert layout['width_cm'] == 19 and layout['font_pt'] == 12
    assert layout['dpi'] == 300 and layout['png_only']
    assert layout['text']
    width, height = 19 / 2.54, layout['height_cm'] / 2.54
    for item in layout['text']:
        assert item['font_pt'] == 12
        x, y, w, h = item['bounds_in']
        assert x >= -.002 and y >= -.002 and x + w <= width + .002 and y + h <= height + .002
    for ext in ['pdf', 'svg']:
        assert not (folder / f'{stem}.{ext}').exists()
    with Image.open(folder / f'{stem}.png') as im:
        assert abs(im.width - width * 300) < 2 and abs(im.height - height * 300) < 2
        assert all(abs(value - 300) < .1 for value in im.info['dpi'])
    return layout
