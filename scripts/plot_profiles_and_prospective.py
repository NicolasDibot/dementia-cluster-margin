"""Draw cluster profiles, stability and prospective validation from aggregates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BLUE, ORANGE, BLACK, GREY = "#2B7BDE", "#F2643B", "#222222", "#666666"
COHORTS = ("adams", "hcap", "share")
METRICS = ("heldout_adjusted_rand", "target_label_agreement", "target_margin_spearman")
OUTCOMES = ("hcap_severe_cognitive_impairment", "informant_assistance_fraction", "delayed_memory_impairment")


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        return data["rows"] if isinstance(data, dict) else data
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(row, names):
    result = {name: float(row[name]) for name in names}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Plotted values must be finite")
    return result


def profiles(rows):
    prepared = []
    for row in rows:
        cohort, label = row["cohort"].lower(), int(row["label"])
        if cohort not in COHORTS or label not in (0, 1):
            raise ValueError("Unknown cohort or cluster label")
        values = numeric(row, ("cognition_mean", "function_mean", "share"))
        if not 0 < values["share"] < 1:
            raise ValueError("Each cluster share must be a fraction strictly between zero and one")
        prepared.append({"cohort": cohort, "label": label, **values,
                         "profile_definition": row["profile_definition"]})
    if len(prepared) != 6 or len({(r["cohort"], r["label"]) for r in prepared}) != 6:
        raise ValueError("Profiles require exactly two clusters in each of the three cohorts")
    for cohort in COHORTS:
        if abs(sum(r["share"] for r in prepared if r["cohort"] == cohort) - 1) > 1e-8:
            raise ValueError("Cluster shares must sum to one within each cohort")
    return prepared


def stability(rows):
    prepared = []
    for row in rows:
        cohort, metric = row["cohort"].lower(), row["metric"]
        if cohort not in COHORTS[:2] or metric not in METRICS:
            raise ValueError("Unknown stability cohort or metric")
        values = numeric(row, ("median", "percentile_10"))
        if not 0 <= values["percentile_10"] <= values["median"] <= 1:
            raise ValueError("Stability summaries must be ordered in [0, 1]")
        prepared.append({"cohort": cohort, "metric": metric, **values})
    if len(prepared) != 6 or len({(r["cohort"], r["metric"]) for r in prepared}) != 6:
        raise ValueError("Stability requires three metrics in both HRS cohorts")
    return prepared


def prospective(rows):
    prepared = []
    for row in rows:
        # A standard pipeline summary also contains paired contrasts.
        if row["estimand"] == "margin_minus_label":
            continue
        outcome, estimand = row["outcome"], row["estimand"]
        if outcome not in OUTCOMES or estimand not in ("label", "margin"):
            raise ValueError("Unknown prospective outcome or estimand")
        values = numeric(row, ("estimate", "ci95_low", "ci95_high"))
        if not 0 <= values["ci95_low"] <= values["estimate"] <= values["ci95_high"] <= 1:
            raise ValueError("Prospective estimates and intervals must be ordered in [0, 1]")
        prepared.append({"outcome": outcome, "estimand": estimand, **values})
    if len(prepared) != 6 or len({(r["outcome"], r["estimand"]) for r in prepared}) != 6:
        raise ValueError("Prospective display requires label and margin for all three outcomes")
    return prepared


class Canvas:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.scale = 2126 / width
        self.image = Image.new("RGB", (2126, round(height * self.scale)), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{180*height/width:.4f}mm" viewBox="0 0 {width} {height}">',
                      '<rect width="100%" height="100%" fill="white"/>']
        self.font_path = next((p for p in (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("C:/Windows/Fonts/arial.ttf")) if p.is_file()), None)

    def text(self, x, y, value, size=13, anchor="start", bold=False, color=BLACK):
        weight = ' font-weight="bold"' if bold else ""
        self.parts.append(f'<text x="{x:.4f}" y="{y:.4f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{color}"{weight}>{html.escape(str(value))}</text>')
        size_px = max(1, round(size * self.scale))
        font = ImageFont.truetype(str(self.font_path), size_px) if self.font_path else ImageFont.load_default(size=size_px)
        self.draw.text((x*self.scale,y*self.scale), str(value), font=font, fill=color,
                       anchor={"start":"ls", "middle":"ms", "end":"rs"}[anchor],
                       stroke_width=0 if not bold else max(1, round(.3*self.scale)))

    def lines(self, x, y, values, **kwargs):
        step = kwargs.get("size", 13) * 1.2
        for i, value in enumerate(values):
            self.text(x, y+i*step, value, **kwargs)

    def line(self, x1, y1, x2, y2, color=BLACK, width=1, dash=False):
        style = ' stroke-dasharray="5 5"' if dash else ""
        self.parts.append(f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}" stroke="{color}" stroke-width="{width}"{style}/>')
        segments = [(0,1)]
        length = math.hypot(x2-x1,y2-y1)
        if dash and length:
            segments = [(i/length,min(i+5,length)/length) for i in range(0,math.ceil(length),10)]
        for a,b in segments:
            self.draw.line(((x1+a*(x2-x1))*self.scale,(y1+a*(y2-y1))*self.scale,
                            (x1+b*(x2-x1))*self.scale,(y1+b*(y2-y1))*self.scale),
                           fill=color,width=max(1,round(width*self.scale)))

    def marker(self, x, y, shape="circle", color=BLACK, radius=5):
        s = self.scale
        if shape == "circle":
            self.parts.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="{radius}" fill="{color}"/>')
            self.draw.ellipse(((x-radius)*s,(y-radius)*s,(x+radius)*s,(y+radius)*s),fill=color)
        elif shape == "square":
            self.parts.append(f'<rect x="{x-radius:.4f}" y="{y-radius:.4f}" width="{2*radius}" height="{2*radius}" fill="{color}"/>')
            self.draw.rectangle(((x-radius)*s,(y-radius)*s,(x+radius)*s,(y+radius)*s),fill=color)
        else:
            points = [(x,y-radius-1),(x-radius-1,y+radius),(x+radius+1,y+radius)]
            self.parts.append('<polygon points="'+' '.join(f'{a:.4f},{b:.4f}' for a,b in points)+f'" fill="{color}"/>')
            self.draw.polygon([(a*s,b*s) for a,b in points],fill=color)

    def group(self, attributes):
        self.parts.append('<g '+' '.join(f'data-{key}="{html.escape(str(value))}"' for key,value in attributes.items())+'>')

    def end_group(self):
        self.parts.append('</g>')

    def save(self, directory, stem):
        (directory/f"{stem}.svg").write_text("\n".join(self.parts+["</svg>"])+"\n")
        self.image.save(directory/f"{stem}.png",dpi=(300,300))


def figure2(profile_rows, stability_rows, directory):
    fig = Canvas(1100,470)
    fig.text(15,25,"a",18,bold=True)
    fig.text(605,25,"b",18,bold=True)
    shapes = {"adams":"circle", "hcap":"triangle", "share":"square"}
    # Axis extent is derived from supplied profiles, never from a target result.
    xhigh = max(2.5, math.ceil(max(r["function_mean"] for r in profile_rows)*2+1)/2)
    x = lambda v: 80+(v+.5)/(xhigh+.5)*440
    y = lambda v: 335-(v+.3)/1.8*285
    fig.line(80,335,520,335); fig.line(80,50,80,335)
    fig.line(x(0),50,x(0),335,"#CCCCCC")
    fig.line(80,y(0),520,y(0),"#CCCCCC")
    for value in range(math.floor(xhigh)+1):
        fig.line(x(value),335,x(value),340)
        fig.text(x(value),358,str(value),12,"middle")
    for value in (0,.5,1.0,1.5):
        fig.line(75,y(value),80,y(value))
        fig.text(70,y(value)+4,f"{value:.1f}",12,"end")
    fig.text(80,43,"Mean cognitive burden (s.d.)",14)
    fig.text(300,382,"Mean functional burden (s.d.)",14,"middle")
    for row in profile_rows:
        fig.group({k:row[k] for k in ("cohort","label","cognition_mean","function_mean","share")})
        px,py = x(row["function_mean"]),y(row["cognition_mean"])
        fig.marker(px,py,shapes[row["cohort"]],ORANGE if row["label"] else GREY,5.2)
        if row["label"]:
            dx,dy = {"adams":(13,20),"hcap":(-15,-19),"share":(12,-15)}[row["cohort"]]
            name = {"adams":"HRS, ADAMS period","hcap":"HRS, HCAP period","share":"SHARE"}[row["cohort"]]
            fig.lines(px+dx,py+dy,[name,f'{100*row["share"]:.1f}%'],size=12,
                      anchor="end" if dx<0 else "start")
        fig.end_group()
    low_shares = [100*r["share"] for r in profile_rows if not r["label"]]
    fig.lines(x(.13),y(-.19),["Non-dementia-like clusters",f"({min(low_shares):.0f}–{max(low_shares):.0f}% of observations)"],size=11,color=GREY)
    lower = min(.98,math.floor(min(r["percentile_10"] for r in stability_rows)*200)/200)
    sx = lambda v: 835+(v-lower)/(1.001-lower)*245
    fig.line(835,335,1080,335)
    step = .005 if lower>=.97 else .01
    for i in range(round((1-lower)/step)+1):
        value=lower+i*step
        fig.line(sx(value),335,sx(value),340)
        fig.text(sx(value),358,f"{value:.3f}",11,"middle")
    names = [["Adjusted Rand index,","held-out participants"],
             ["Label agreement,","assessed participants"],
             ["Margin rank correlation,","assessed participants"]]
    positions = [85,185,285]
    for i,name in enumerate(names):
        fig.lines(822,positions[i]-5,name,size=12,anchor="end")
    for row in stability_rows:
        py=positions[METRICS.index(row["metric"])]+(-12 if row["cohort"]=="hcap" else 12)
        fig.group(row)
        fig.line(sx(row["percentile_10"]),py,sx(row["median"]),py,GREY,1.6)
        fig.marker(sx(row["median"]),py,shapes[row["cohort"]],BLACK,4.8)
        fig.end_group()
    fig.text(957,383,"Median across refits",13,"middle")
    fig.text(957,399,"Bar starts at the 10th percentile",11,"middle")
    for cx,cohort,label in [(100,"adams","HRS, ADAMS period"),(335,"hcap","HRS, HCAP period"),(570,"share","SHARE")]:
        fig.marker(cx,437,shapes[cohort],BLACK,4.5);fig.text(cx+13,441,label,12)
    fig.save(directory,"Figure_2")


def figure4(rows,directory):
    fig = Canvas(1100,460)
    low=min(.49,math.floor(min(r["ci95_low"] for r in rows)*20)/20)
    high=max(.72,math.ceil(max(r["ci95_high"] for r in rows)*20)/20)
    x=lambda v:360+(v-low)/(high-low)*660
    fig.line(360,360,1020,360)
    fig.line(x(.5),35,x(.5),360,GREY,1,True)
    labels=[["Probable dementia","(AUC)"],["Informant-rated functional","limitation (concordance)"],["Delayed recall","(concordance)"]]
    positions=[90,205,315]
    for py,label in zip(positions,labels):
        fig.lines(345,py-5,label,size=16,anchor="end")
    first=math.ceil(low*20)/20
    for i in range(math.floor((high-first)*20+1e-7)+1):
        value=first+i*.05
        fig.line(x(value),360,x(value),366)
        fig.text(x(value),389,f"{value:.2f}",14,"middle")
    for row in rows:
        py=positions[OUTCOMES.index(row["outcome"])]+(-14 if row["estimand"]=="margin" else 14)
        color=BLUE if row["estimand"]=="margin" else ORANGE
        fig.group(row)
        fig.line(x(row["ci95_low"]),py,x(row["ci95_high"]),py,color,2)
        fig.marker(x(row["estimate"]),py,color=color,radius=5.5)
        fig.end_group()
    fig.text(690,416,"Within-country AUC or concordance",16,"middle")
    fig.marker(495,443,color=ORANGE,radius=5);fig.text(510,448,"Dementia-like label",14)
    fig.marker(720,443,color=BLUE,radius=5);fig.text(735,448,"Margin",14)
    fig.save(directory,"Figure_4")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("profiles","stability","prospective"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    if bool(args.profiles)!=bool(args.stability):
        parser.error("Figure 2 requires both --profiles and --stability")
    if not args.profiles and not args.prospective:
        parser.error("Supply the inputs for Figure 2, Figure 4, or both")
    output=args.output_dir.expanduser().resolve()
    repository=Path(__file__).resolve().parents[1]
    if output==repository or repository in output.parents:
        parser.error("Generated figures and source values must be outside the repository")
    output.mkdir(parents=True,exist_ok=True)
    report={"source_sha256":{},"figures":[]}
    for name in ("profiles","stability","prospective"):
        path=getattr(args,name)
        if path:
            report["source_sha256"][name]=hashlib.sha256(path.read_bytes()).hexdigest()
    if args.profiles:
        pr,st=profiles(read_rows(args.profiles)),stability(read_rows(args.stability))
        figure2(pr,st,output)
        (output/"Source_Data_Figure_2.json").write_text(json.dumps({"profiles":pr,"stability":st},indent=2)+"\n")
        report["figures"].append({"figure":2,"profile_points":len(pr),"stability_points":len(st)})
    if args.prospective:
        rows=prospective(read_rows(args.prospective))
        figure4(rows,output)
        (output/"Source_Data_Figure_4.json").write_text(json.dumps({"rows":rows},indent=2)+"\n")
        report["figures"].append({"figure":4,"estimates":len(rows),"confidence_bounds":2*len(rows)})
    (output/"figure_checks.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
