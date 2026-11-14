"""Plot existing estimates exactly; no models or inferential statistics are refitted."""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import math
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', required=True, type=Path,
                    help='Paired aggregate model estimates in CSV format.')
parser.add_argument('--output-dir', required=True, type=Path)
args = parser.parse_args()
ROOT = args.output_dir
ROOT.mkdir(parents=True, exist_ok=True)
SOURCE = args.input

def load_font(size, bold=False):
    candidates = (
        ['Arial Bold.ttf', 'DejaVuSans-Bold.ttf',
         '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
         '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'] if bold else
        ['Arial.ttf', 'DejaVuSans.ttf',
         '/System/Library/Fonts/Supplemental/Arial.ttf',
         '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)
with SOURCE.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
ORDER = [
    ("less_education", "Less education"), ("hearing_loss", "Hearing loss"),
    ("poor_vision", "Poor vision"), ("depression", "Depression"),
    ("physical_inactivity", "Physical inactivity"), ("diabetes", "Diabetes"),
    ("hypertension", "Hypertension"), ("high_cholesterol", "High cholesterol"),
    ("smoking", "Smoking"), ("obesity", "Obesity"),
    ("excessive_alcohol", "Excessive alcohol"), ("no_social_activity", "No social activity"),
]
by_key = {(row["cohort"], row["factor"]): row for row in rows}
assert len(rows) == len(by_key) == 23
assert ("HRS", "high_cholesterol") not in by_key
assert sum(row["cohort"] == "HRS" for row in rows) == 11
assert sum(row["cohort"] == "SHARE" for row in rows) == 12
for row in rows:
    assert float(row["cluster_entry_n"]) == float(row["margin_change_n"])
    assert float(row["cluster_entry_entries"]) == float(row["margin_change_entries"])

SCALE = 3
WIDTH, HEIGHT = 1020, 480
image = Image.new("RGB", (WIDTH * SCALE, HEIGHT * SCALE), "white")
draw = ImageDraw.Draw(image)
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">', '<rect width="100%" height="100%" fill="white"/>']

def line(x1, y1, x2, y2, color="#333333", width=0.85, dash=False):
    if dash:
        assert x1 == x2
        y = y1
        while y < y2:
            draw.line((x1*SCALE, y*SCALE, x2*SCALE, min(y+4,y2)*SCALE), fill=color, width=max(1,round(width*SCALE)))
            y += 8
    else:
        draw.line((x1*SCALE, y1*SCALE, x2*SCALE, y2*SCALE), fill=color, width=max(1,round(width*SCALE)))
    extra = ' stroke-dasharray="4 4"' if dash else ''
    svg.append(f'<line x1="{x1:.6f}" y1="{y1:.6f}" x2="{x2:.6f}" y2="{y2:.6f}" stroke="{color}" stroke-width="{width}"{extra}/>')

def text(x, y, value, size=12, anchor="start", bold=False):
    font = load_font(round(size*SCALE), bold)
    pil_anchor = {"start":"lm", "middle":"mm", "end":"rm"}[anchor]
    draw.text((x*SCALE,y*SCALE), value, font=font, fill="black", anchor=pil_anchor)
    weight = ' font-weight="bold"' if bold else ''
    svg.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" dominant-baseline="central" fill="black"{weight}>{html.escape(value)}</text>')

def circle(x, y, color, filled, radius=3.8):
    fill = color if filled else "white"
    draw.ellipse(((x-radius)*SCALE,(y-radius)*SCALE,(x+radius)*SCALE,(y+radius)*SCALE), fill=fill, outline=color, width=round(1.2*SCALE))
    svg.append(f'<circle cx="{x:.6f}" cy="{y:.6f}" r="{radius}" fill="{fill}" stroke="{color}" stroke-width="1.2"/>')

panels = [
    ("margin_change", "#2B7BDE", 155, 490, -0.22, 0.68, 0.0, False,
     "Difference in standardized change", "of the margin (s.d.)"),
    ("cluster_entry", "#F2643B", 650, 985, 0.3, 3.2, 1.0, True,
     "Odds ratio of entering the", "dementia-like cluster (label change)"),
]
plotted = []
for panel_idx, (prefix, color, left, right, low_axis, high_axis, null, logarithmic, xlabel1, xlabel2) in enumerate(panels):
    def xpos(value):
        if logarithmic:
            return left + (math.log(value)-math.log(low_axis))/(math.log(high_axis)-math.log(low_axis))*(right-left)
        return left + (value-low_axis)/(high_axis-low_axis)*(right-left)
    line(left,24,left,380)
    line(left,380,right,380)
    line(xpos(null),24,xpos(null),380,color="#7A7A7A",width=0.8,dash=True)
    text(left-110,13,"ab"[panel_idx],size=16,bold=True)
    for idx, (factor, label) in enumerate(ORDER):
        y = 38 + idx*29.3
        text(left-7,y,label,size=12,anchor="end")
        line(left-2.8,y,left,y,width=0.7)
        for cohort, offset, filled in [("HRS", 4, True), ("SHARE", -4, False)]:
            row = by_key.get((cohort, factor))
            if row is None:
                continue
            estimate = float(row[f"{prefix}_estimate"])
            low = float(row[f"{prefix}_ci95_low"])
            high = float(row[f"{prefix}_ci95_high"])
            assert all(math.isfinite(v) for v in (estimate, low, high))
            assert low_axis <= low <= estimate <= high <= high_axis
            line(xpos(low),y+offset,xpos(high),y+offset,color=color,width=1.25)
            circle(xpos(estimate),y+offset,color,filled)
            plotted.append({"panel": prefix,"cohort":cohort,"factor":factor,
                            "estimate":estimate,"ci95_low":low,"ci95_high":high})
    ticks = [(0.3,"0.3"),(1,"1.0"),(3,"3.0")] if logarithmic else [(-0.2,"−0.2"),(0,"0.0"),(0.2,"0.2"),(0.4,"0.4"),(0.6,"0.6")]
    for value, label in ticks:
        x = xpos(value)
        line(x,380,x,383.2,width=0.7)
        text(x,394,label,size=11.5,anchor="middle")
    text((left+right)/2,414,xlabel1,size=14,anchor="middle")
    text((left+right)/2,432,xlabel2,size=14,anchor="middle")

circle(530,462,"black",True,radius=3.8)
text(541,462,"HRS",size=12)
circle(590,462,"black",False,radius=3.8)
text(601,462,"SHARE",size=12)
svg.append("</svg>")
assert len(plotted) == 46
for panel, *_ in panels:
    assert sum(row["panel"] == panel for row in plotted) == 23
    assert not any(row["panel"] == panel and row["cohort"] == "HRS" and row["factor"] == "high_cholesterol" for row in plotted)
image.save(ROOT / "Figure5_common_population.png", dpi=(300,300))
(ROOT / "Figure5_common_population.svg").write_text("\n".join(svg)+"\n",encoding="utf-8")
report = {
    "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "source_rows":len(rows), "points_per_panel":23,
    "HRS_points_per_panel":11, "SHARE_points_per_panel":12,
    "HRS_cholesterol":"not available", "plotted_values":plotted,
}
(ROOT / "Figure5_common_population_plot_check.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps({key:value for key,value in report.items() if key!="plotted_values"},indent=2))
print(ROOT / "Figure5_common_population.png")
