#!/usr/bin/env python3
"""Monochrome C SPRING/WSSR trajectories with a full 10k trailing window."""
import csv, html, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = -37.84471
WINDOW = 10000
RUNS = [
    ("SPRING, lr=0.02", Path("/scratch/dexuan1/runs/e40000/C/spring_lr02"), "", "circle", 3.2),
    ("WSSR control, eta=0.2, lr=0.002, SVD 8/2", Path("/scratch/dexuan1/runs/e20000_c_focus/C/C_wssr_control_eta02_lr0002_svd8_2"), "10 4", "square", 2.4),
    ("WSSR eta=0.2, lr=0.02, SVD 8/5", Path("/scratch/dexuan1/runs/e20000_c_focus/C/C_wssr_eta02_lr02_svd8_5"), "3 3", "triangle", 2.2),
    ("WSSR eta=0.5, lr=0.02, SVD 8/5", Path("/scratch/dexuan1/runs/e20000_c_focus/C/C_wssr_eta05_lr02_svd8_5"), "12 4 2 4", "diamond", 2.2),
    ("WSSR eta=0.8, lr=0.02, SVD 8/5", Path("/scratch/dexuan1/runs/e20000_c_focus/C/C_wssr_eta08_lr02_svd8_5"), "6 3 1 3", "plus", 2.2),
    ("WSSR eta=0.9, lr=0.02, SVD 8/5", Path("/scratch/dexuan1/runs/e20000_c_focus/C/C_wssr_eta09_lr02_svd8_5"), "2 3", "cross", 2.2),
]

def read_metrics(path):
    out=[]
    with (path/"training_metrics.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            try: out.append((float(row["energy"]),float(row["variance"])))
            except (ValueError,KeyError): pass
    return out

def rolling_full(values):
    total=sum(values[:WINDOW]); out=[(WINDOW,total/WINDOW)]
    for i in range(WINDOW,len(values)):
        total += values[i]-values[i-WINDOW]
        out.append((i+1,total/WINDOW))
    return out

def marker(kind,x,y):
    if kind=="circle": return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="white" stroke="black" stroke-width="1.4"/>'
    if kind=="square": return f'<rect x="{x-3:.1f}" y="{y-3:.1f}" width="6" height="6" fill="white" stroke="black" stroke-width="1.4"/>'
    if kind=="triangle": return f'<path d="M{x:.1f},{y-3.8:.1f} L{x-3.5:.1f},{y+3:.1f} L{x+3.5:.1f},{y+3:.1f} Z" fill="white" stroke="black" stroke-width="1.3"/>'
    if kind=="diamond": return f'<path d="M{x:.1f},{y-4:.1f} L{x-4:.1f},{y:.1f} L{x:.1f},{y+4:.1f} L{x+4:.1f},{y:.1f} Z" fill="white" stroke="black" stroke-width="1.3"/>'
    if kind=="plus": return f'<path d="M{x-4:.1f},{y:.1f} L{x+4:.1f},{y:.1f} M{x:.1f},{y-4:.1f} L{x:.1f},{y+4:.1f}" stroke="black" stroke-width="1.5"/>'
    return f'<path d="M{x-3.5:.1f},{y-3.5:.1f} L{x+3.5:.1f},{y+3.5:.1f} M{x-3.5:.1f},{y+3.5:.1f} L{x+3.5:.1f},{y-3.5:.1f}" stroke="black" stroke-width="1.5"/>'

def plot(series,title,ylabel,out):
    W,H=1180,660; l,r,t,b=92,390,48,72; x0,x1,y0,y1=l,W-r,t,H-b
    pts=[p for s in series for p in s[1]]; xmin,xmax=WINDOW,max(x for x,y in pts); ymin,ymax=min(y for x,y in pts),max(y for x,y in pts)
    pad=.06*(ymax-ymin or 1); ymin-=pad; ymax+=pad
    sx=lambda x:x0+(x-xmin)*(x1-x0)/(xmax-xmin); sy=lambda y:y1-(y-ymin)*(y1-y0)/(ymax-ymin)
    q=[f'<rect width="{W}" height="{H}" fill="white"/>',f'<text x="{(x0+x1)/2}" y="27" text-anchor="middle" font-size="18">{html.escape(title)}</text>',f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="black"/>',f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="black"/>']
    for i in range(6):
        f=i/5; xv=xmin+f*(xmax-xmin); yv=ymin+f*(ymax-ymin); xx,yy=sx(xv),sy(yv)
        q += [f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" stroke="#ddd"/>',f'<text x="{xx:.1f}" y="{y1+22}" text-anchor="middle" font-size="11">{xv:.0f}</text>',f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#ddd"/>',f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{yv:.3g}</text>']
    q += [f'<text x="{(x0+x1)/2}" y="{H-20}" text-anchor="middle" font-size="14">Iteration</text>',f'<text x="20" y="{(y0+y1)/2}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {(y0+y1)/2})">{html.escape(ylabel)}</text>']
    for i,(label,points,dash,kind,width) in enumerate(series):
        coords=' '.join(f'{sx(x):.2f},{sy(y):.2f}' for x,y in points)
        da=f' stroke-dasharray="{dash}"' if dash else ''
        q.append(f'<polyline points="{coords}" fill="none" stroke="black" stroke-width="{width}"{da}/>')
        for x,y in points:
            if x%2000==0: q.append(marker(kind,sx(x),sy(y)))
        ly=y0+18+i*40; q += [f'<line x1="{x1+20}" y1="{ly}" x2="{x1+55}" y2="{ly}" stroke="black" stroke-width="{width}"{da}/>',marker(kind,x1+37.5,ly),f'<text x="{x1+64}" y="{ly+4}" font-size="10.5">{html.escape(label)}</text>']
    q.append(f'<text x="{x1+20}" y="{y1-12}" font-size="10" fill="#555">Full trailing window = 10,000 iterations</text>')
    out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif">{"".join(q)}</svg>\n')

def main():
    energy=[]; variance=[]
    for label,path,dash,kind,width in RUNS:
        rows=read_metrics(path)
        e=[x[0] for x in rows]; v=[x[1] for x in rows]
        energy.append((label,[(x,(y-REF)*1000) for x,y in rolling_full(e)],dash,kind,width))
        variance.append((label,rolling_full(v),dash,kind,width))
    figs=HERE/"report_figs"; figs.mkdir(exist_ok=True)
    plot(energy,"C energy-error trajectories","10,000-iteration mean error (mHa)",figs/"C_focus_energy_error_window10000.svg")
    plot(variance,"C variance trajectories","10,000-iteration mean variance (Ha²)",figs/"C_focus_variance_window10000.svg")
    print('wrote two figures')
if __name__=="__main__": main()
