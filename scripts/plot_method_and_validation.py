"""Draw the method schematic and clinical-validation figure as editable SVGs."""

from pathlib import Path
import argparse
import csv
import html
import json
import math
import random

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True,
                    help='Directory for generated figures and aggregate source values.')
parser.add_argument('--clinical-estimates', type=Path,
                    help='Long aggregate CSV with cohort, estimand, estimate and confidence limits.')
args = parser.parse_args()
ROOT = args.output_dir.expanduser().resolve()
REPOSITORY = Path(__file__).resolve().parents[1]
if ROOT == REPOSITORY or REPOSITORY in ROOT.parents:
    parser.error('Generated figures and source values must be outside the repository')
ROOT.mkdir(parents=True, exist_ok=True)
BLUE = '#2B7BDE'
ORANGE = '#F2643B'
BLACK = '#222222'


class SVG:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="180mm" height="{180*height/width:.4f}mm" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']

    def text(self, x, y, content, size=13, anchor='start', bold=False, color=BLACK, rotate=None):
        extra = ' font-weight="bold"' if bold else ''
        if rotate is not None:
            extra += f' transform="rotate({rotate} {x} {y})"'
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}" fill="{color}"{extra}>{html.escape(content)}</text>')

    def lines(self, x, y, strings, size=13, anchor='start', bold=False, color=BLACK, step=None):
        for i, string in enumerate(strings):
            self.text(x, y+i*(step or size*1.2), string, size, anchor, bold, color)

    def line(self, x1, y1, x2, y2, color=BLACK, width=1, dash=False):
        extra = ' stroke-dasharray="6 5"' if dash else ''
        self.parts.append(f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def rect(self, x, y, w, h, fill):
        self.parts.append(f'<rect x="{x:.4f}" y="{y:.4f}" width="{w:.4f}" height="{h:.4f}" fill="{fill}"/>')

    def circle(self, x, y, radius, fill=BLACK, stroke=None, width=1):
        self.parts.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="{radius}" fill="{fill}" stroke="{stroke or fill}" stroke-width="{width}"/>')

    def path(self, points, stroke, width=1):
        d = ' '.join(('M' if i == 0 else 'L')+f'{x:.4f},{y:.4f}' for i,(x,y) in enumerate(points))
        self.parts.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}"/>')

    def arrow(self, x1, y1, x2, y2, color=BLACK, width=2):
        self.line(x1,y1,x2,y2,color,width)
        angle = math.atan2(y2-y1,x2-x1)
        tip = [(x2,y2), (x2-8*math.cos(angle-.5),y2-8*math.sin(angle-.5)), (x2-8*math.cos(angle+.5),y2-8*math.sin(angle+.5))]
        self.parts.append('<polygon points="'+ ' '.join(f'{x},{y}' for x,y in tip)+f'" fill="{color}"/>')

    def save(self, path):
        path.write_text('\n'.join(self.parts+['</svg>'])+'\n')


def clinical_values(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    mapping = {
        'label': 'auc::label', 'margin': 'auc::margin',
        'gain': 'contrast::margin_minus_label',
        'outside': 'contribution::two_cluster::H0',
        'inside': 'contribution::two_cluster::H1',
        'label_or': 'or::label', 'margin_or': 'or::margin',
    }
    result = {}
    for cohort in ('ADAMS', 'HCAP'):
        values = {}
        for name, estimand in mapping.items():
            match = [r for r in rows if r['cohort'].upper() == cohort and r['estimand'] == estimand]
            if len(match) != 1:
                raise ValueError(f'Expected one {cohort} {estimand} estimate')
            row = match[0]
            triplet = tuple(float(row[k]) for k in ('estimate', 'ci95_low', 'ci95_high'))
            if not all(math.isfinite(v) for v in triplet) or not triplet[1] <= triplet[0] <= triplet[2]:
                raise ValueError('Clinical estimate or interval is invalid')
            values[name] = triplet[0] if name in ('outside', 'inside') else triplet
        gain = values['gain'][0]
        if gain <= 0 or values['outside'] < 0 or values['inside'] < 0:
            raise ValueError('The stacked display requires a positive gain and nonnegative components')
        if not math.isclose(values['outside'] + values['inside'], gain, abs_tol=1e-8):
            raise ValueError('Clinical components do not sum to the total AUC gain')
        if not math.isclose(values['margin'][0] - values['label'][0], gain, abs_tol=1e-8):
            raise ValueError('Clinical AUC contrast is inconsistent with the supplied AUCs')
        if min(values['label_or'] + values['margin_or']) <= 0:
            raise ValueError('Odds ratios and their intervals must be positive')
        values['share'] = f"{100 * values['outside'] / gain:.1f}%"
        result[cohort] = values
    return result


def clinical_figure():
    fig = SVG(1100, 360)
    values = {
        'ADAMS': {'label': (.726,.677,.778), 'margin': (.855,.821,.888),
                  'gain': (.130,.090,.164), 'outside': .119, 'inside': .011,
                  'share': '91.6%', 'label_or': (12.8,6.2,26.3), 'margin_or': (2.06,1.49,2.85)},
        'HCAP': {'label': (.554,.532,.580), 'margin': (.712,.672,.750),
                 'gain': (.157,.122,.192), 'outside': .157, 'inside': .0004,
                 'share': '99.8%', 'label_or': (3.48,1.97,6.16), 'margin_or': (1.53,1.31,1.79)},
    }
    if args.clinical_estimates:
        values = clinical_values(args.clinical_estimates)
    panels = [(140,330,'a'), (520,710,'b'), (905,1090,'c')]
    for left,right,letter in panels:
        fig.text(left-120,22,letter,18,bold=True)
        fig.line(left,250,right,250,width=.9)
        for cohort,y in [('ADAMS',85),('HCAP',190)]:
            label = 'consensus dementia' if cohort=='ADAMS' else 'algorithmic dementia'
            fig.lines(left-10,y-4,[cohort,label],12,'end')
    xauc = lambda v:140+(v-.5)/.5*190
    for v in [.5,.6,.7,.8,.9,1.]:
        x=xauc(v); fig.line(x,250,x,254,width=.8); fig.text(x,272,f'{v:.1f}',11,'middle')
    fig.line(xauc(.5),38,xauc(.5),250,'#777777',.8,True)
    xgain=lambda v:520+v/.25*190
    for v in [0,.05,.10,.15,.20,.25]:
        x=xgain(v); fig.line(x,250,x,254,width=.8); fig.text(x,272,f'{v:.2f}',11,'middle')
    xor=lambda v:905+(math.log(v)-math.log(.8))/(math.log(35)-math.log(.8))*185
    for v in [1,2,5,10,20]:
        x=xor(v); fig.line(x,250,x,254,width=.8); fig.text(x,272,str(v),11,'middle')
    fig.line(xor(1),38,xor(1),250,'#777777',.8,True)
    for cohort,y in [('ADAMS',85),('HCAP',190)]:
        d=values[cohort]
        for kind,color,offset in [('margin',BLUE,-11),('label',ORANGE,11)]:
            point,lo,hi=d[kind]
            fig.line(xauc(lo),y+offset,xauc(hi),y+offset,color,1.4)
            fig.circle(xauc(point),y+offset,4,color)
            point,lo,hi=d[kind+'_or']
            fig.line(xor(lo),y+offset,xor(hi),y+offset,color,1.4)
            fig.circle(xor(point),y+offset,4,color)
        fig.rect(xgain(0),y-18,xgain(d['outside'])-xgain(0),36,BLUE)
        fig.rect(xgain(d['outside']),y-18,xgain(d['outside']+d['inside'])-xgain(d['outside']),36,ORANGE)
        point,lo,hi=d['gain']
        fig.line(xgain(lo),y,xgain(hi),y,BLACK,1.5)
        fig.circle(xgain(point),y,3.5)
        fig.text(xgain(hi)+7,y+4,d['share'],11)
    fig.text(235,294,'Survey-weighted AUC',13,'middle')
    fig.text(615,294,'AUC gain, margin minus label',13,'middle')
    fig.text(997,294,'Adjusted odds ratio',13,'middle')
    fig.text(997,310,'(margin per s.d.)',12,'middle')
    fig.circle(140,329,4,ORANGE); fig.text(150,333,'Dementia-like label',12)
    fig.circle(285,329,4,BLUE); fig.text(295,333,'Margin',12)
    fig.rect(520,316,11,11,ORANGE); fig.text(539,326,'Gain from pairs inside the cluster',11.5)
    fig.rect(520,337,11,11,BLUE); fig.text(539,347,'Gain from pairs outside the cluster',11.5)
    fig.save(ROOT/'Figure_3.svg')
    return values


def schematic_figure():
    fig=SVG(1100,1050)
    fig.text(20,28,'a',20,bold=True)
    boxes=[(90,['29 survey measures'],['memory, mobility,','daily activities']),
           (330,['Multiple factor analysis'],['5 summary dimensions']),
           (570,['Ward + k-means'],['two clusters, two centroids']),
           (810,['Label and margin'],['H = 1[M > 0]','M = d(N) − d(D)'])]
    for idx,(x,title,sub) in enumerate(boxes):
        fig.rect(x,40,200,75,'#F1F0ED')
        fig.lines(x+100,67,title,16,'middle',True)
        fig.lines(x+100,89,sub,13,'middle',color='#555555')
        if idx<3:fig.arrow(x+204,77,x+236,77)
    fig.text(550,139,'Reference classifications are never used to fit the clusters or define the margin',14,'middle',color='#555555')
    fig.text(20,190,'b',20,bold=True);fig.text(580,190,'c',20,bold=True)
    fig.text(50,210,'The label cuts a continuum',16,bold=True)
    fig.text(50,229,'The boundary lies halfway between the two centroids (+)',12.5,color='#555555')
    fig.text(610,210,'The margin: M = d(N) − d(D)',16,bold=True)
    fig.text(610,229,'Gray curves join points of equal margin; dashed line: M = 0',12.5,color='#555555')
    # Both panels use the same simulated cloud and the same centroids.
    rng=random.Random(7417)
    points=[]
    while len(points)<550:
        u=rng.gauss(0,1)
        x=-.15+.95*u
        y=.03+.50*u+.40*rng.gauss(0,1)
        if -2.0<x<2.7 and -1.55<y<1.95:points.append((x,y))
    centres=[(-.55,-.30),(1.25,.75)]
    for iteration in range(100):
        groups=[[],[]]
        for x,y in points:
            label=int(math.hypot(x-centres[1][0],y-centres[1][1]) < math.hypot(x-centres[0][0],y-centres[0][1]))
            groups[label].append((x,y))
        updated=[tuple(sum(p[j] for p in g)/len(g) for j in range(2)) for g in groups]
        if updated==centres:break
        centres=updated
    mid=tuple((a+b)/2 for a,b in zip(*centres))
    dx,dy=(centres[1][i]-centres[0][i] for i in range(2))
    sep=math.hypot(dx,dy); ux,uy=dx/sep,dy/sep; vx,vy=-uy,ux
    margin=lambda x,y:math.hypot(x-centres[0][0],y-centres[0][1])-math.hypot(x-centres[1][0],y-centres[1][1])
    def xy(x,y,left):return left+(x+2)/4.7*460, 620-(y+1.55)/3.5*360
    def color(m):
        low=(43,123,222); zero=(218,216,211); high=(242,100,59)
        p=min(1,abs(m)/sep); aa,bb=(zero,high) if m>=0 else (zero,low)
        return '#'+''.join(f'{round(a+(b-a)*p):02x}' for a,b in zip(aa,bb))
    for side,left in [('b',50),('c',610)]:
        clip=f'clip_{side}'
        fig.parts.append(f'<defs><clipPath id="{clip}"><rect x="{left}" y="250" width="460" height="380"/></clipPath></defs>')
        fig.parts.append(f'<g clip-path="url(#{clip})">')
        if side=='c':
            for fraction in [-.8,-.5,-.2,.2,.5,.8]:
                m=fraction*sep
                aa=abs(m)/2; bb=math.sqrt((sep/2)**2-aa*aa)
                curve=[]
                for k in range(241):
                    v=-3+6*k/240;u=math.copysign(aa*math.sqrt(1+v*v/(bb*bb)),m)
                    curve.append(xy(mid[0]+u*ux+v*vx,mid[1]+u*uy+v*vy,left))
                fig.path(curve,'#AAAAAA',.9)
        for x,y in points:
            m=margin(x,y)
            fig.circle(*xy(x,y,left),2.4,BLUE if side=='b' and m<=0 else ORANGE if side=='b' else color(m))
        bx1,by1=xy(mid[0]-4*vx,mid[1]-4*vy,left)
        bx2,by2=xy(mid[0]+4*vx,mid[1]+4*vy,left)
        fig.line(bx1,by1,bx2,by2,BLACK,1.6,True)
        for c in centres:
            x,y=xy(*c,left);fig.line(x-12,y,x+12,y,BLACK,3);fig.line(x,y-12,x,y+12,BLACK,3)
        if side=='c':
            px,py=xy(.70,1.10,left)
            for c,label,offset in [(centres[0],'d(N)',-10),(centres[1],'d(D)',13)]:
                x,y=xy(*c,left);fig.arrow(px,py,x,y,BLACK,1.7)
                tx,ty=(px+x)/2,(py+y)/2
                fig.rect(tx-17,ty+offset-12,36,17,'white');fig.text(tx,ty+offset,label,13,'middle')
            fig.circle(px,py,6,'white',BLACK,1.8)
        fig.parts.append('</g>')
        fig.text(left+230,650,'Factor dimension 1',15,'middle')
        fig.text(left-13,440,'Factor dimension 2',15,'middle',rotate=-90)
    fig.circle(115,694,5,BLUE);fig.text(130,699,'Non-dementia-like (H = 0)',13)
    fig.circle(335,694,5,ORANGE);fig.text(350,699,'Dementia-like (H = 1)',13)
    fig.text(840,681,'Margin M',14,'middle')
    fig.parts.append('<defs><linearGradient id="margin_gradient"><stop offset="0" stop-color="#2B7BDE"/><stop offset=".5" stop-color="#DAD8D3"/><stop offset="1" stop-color="#F2643B"/></linearGradient></defs>')
    fig.rect(705,694,270,9,'url(#margin_gradient)')
    for x,t in [(705,'< 0'),(840,'0'),(975,'> 0')]:fig.text(x,720,t,12,'middle')
    fig.text(20,769,'d',20,bold=True)
    fig.text(50,789,'Where the margin can add information',16,bold=True)
    fig.rect(215,815,290,173,'#E7F0FA');fig.rect(505,815,290,173,'#FCF0E9')
    fig.line(505,815,505,988,BLACK,1.6,True)
    fig.text(222,835,'H = 0',14,bold=True,color=BLUE);fig.text(790,835,'H = 1',14,'end',True,ORANGE)
    fig.text(190,863,'Dementia',13,'end');fig.text(190,952,'No dementia',13,'end')
    cases=[390,460,545,580,610,645,685,714]
    controls=[248,274,300,325,350,371,390,409,437,455,487,533,585]
    for x in cases:fig.circle(x,859,4.5)
    for x in controls:fig.circle(x,947,5,'white',BLACK,1.5)
    for start,end,num in [(300,460,'1'),(371,645,'2')]:
        fig.line(start,947,end,859,'#666666',1.8)
        tx=(start+end)/2;fig.circle(tx,903,9,'white',BLACK,1);fig.text(tx,907,num,12,'middle')
    fig.lines(815,861,['Pair 1, same label: the label','cannot order the pair,','but the margin can.'],12.5)
    fig.lines(815,938,['Pair 2, opposite labels:','label and margin order the','pair the same way.'],12.5)
    fig.text(505,1019,'AUC(margin) − AUC(label) = gain from pairs within H = 0 + gain from pairs within H = 1',14,'middle',True)
    fig.text(505,1043,'Margin M',14,'middle')
    fig.save(ROOT/'Figure_1.svg')


clinical=clinical_figure()
schematic_figure()
(ROOT/'Source_Data_Figure_3.json').write_text(json.dumps(clinical,indent=2)+'\n')
