#!/usr/bin/env python3
"""Create completed-only E5000 tables and E500/E5000 log-error overlays."""
import csv, html, math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = {"C": -37.84471, "N2eq": -109.5388}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#8C564B", "#17BECF", "#7F7F7F"]

def read(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def roll(values, n=50):
    out=[]; buf=[]; total=0.0
    for v in values:
        buf.append(v); total += v
        if len(buf)>n: total -= buf.pop(0)
        out.append(total/len(buf))
    return out

def svg(system, series, path):
    W,H=1120,650; l,r,t,b=90,280,45,70; x0,x1,y0,y1=l,W-r,t,H-b
    pts=[p for s in series for p in s['points']]
    xmin,xmax=min(x for x,y in pts),max(x for x,y in pts)
    ymin,ymax=min(y for x,y in pts),max(y for x,y in pts); pad=.04*(ymax-ymin); ymin-=pad; ymax+=pad
    sx=lambda x:x0+(x-xmin)*(x1-x0)/(xmax-xmin); sy=lambda y:y1-(y-ymin)*(y1-y0)/(ymax-ymin)
    q=[f'<rect width="{W}" height="{H}" fill="white"/>',f'<text x="{(x0+x1)/2}" y="27" text-anchor="middle" font-size="18">{system}: E500 vs completed E5000 log-error</text>',f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#222"/>',f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222"/>']
    for i in range(6):
        f=i/5; xv=xmin+f*(xmax-xmin); yv=ymin+f*(ymax-ymin); xx,yy=sx(xv),sy(yv)
        q += [f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" stroke="#e5e5e5"/>',f'<text x="{xx:.1f}" y="{y1+21}" text-anchor="middle" font-size="11">{xv:.0f}</text>',f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#e5e5e5"/>',f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{yv:.2f}</text>']
    q += [f'<text x="{(x0+x1)/2}" y="{H-18}" text-anchor="middle" font-size="14">Epoch</text>',f'<text x="20" y="{(y0+y1)/2}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {(y0+y1)/2})">log10(E - E_ref) [Ha]</text>']
    names=sorted(set(s['name'] for s in series)); cmap={n:COLORS[i%len(COLORS)] for i,n in enumerate(names)}
    for s in sorted(series,key=lambda z:(z['version']=='E5000',z['name'])):
        color=cmap[s['name']]; thick=3.0 if s['version']=='E5000' else 1.0; opacity=1 if s['version']=='E5000' else .28
        coords=' '.join(f'{sx(x):.2f},{sy(y):.2f}' for x,y in s['points'])
        q.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{thick}" opacity="{opacity}"/>')
    for i,n in enumerate(names):
        ly=y0+16+i*20; color=cmap[n]
        q += [f'<line x1="{x1+20}" y1="{ly}" x2="{x1+43}" y2="{ly}" stroke="{color}" stroke-width="3"/>',f'<text x="{x1+49}" y="{ly+4}" font-size="11">{html.escape(n)}</text>']
    ly=y0+16+len(names)*20+12
    q += [f'<line x1="{x1+20}" y1="{ly}" x2="{x1+43}" y2="{ly}" stroke="#555" stroke-width="1" opacity=".28"/>',f'<text x="{x1+49}" y="{ly+4}" font-size="11">E500 (thin, pale)</text>',f'<line x1="{x1+20}" y1="{ly+20}" x2="{x1+43}" y2="{ly+20}" stroke="#555" stroke-width="3"/>',f'<text x="{x1+49}" y="{ly+24}" font-size="11">E5000 completed (thick)</text>']
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif">{"".join(q)}</svg>\n')

def main():
    s500=read(HERE/'results/summary.csv'); c500=read(HERE/'results/curves.csv')
    s5k=read(HERE/'results/e5000/summary.csv'); c5k=read(HERE/'results/e5000/curves.csv')
    ok500={(r['system'],r['run_name']) for r in s500 if r['status']=='complete'}
    ok5k={(r['system'],r['run_name']) for r in s5k if r['status']=='complete'}
    series=[]
    for version,rows,ok in [('E500',c500,ok500),('E5000',c5k,ok5k)]:
        g=defaultdict(list)
        for r in rows:
            k=(r['system'],r['run_name'])
            if k in ok and math.isfinite(float(r['energy'])): g[k].append((int(float(r['epoch'])),float(r['energy'])))
        for (system,name),v in g.items():
            v.sort(); sm=roll([e for _,e in v]); pts=[(x,math.log10(e-REF[system])) for (x,_),e in zip(v,sm) if e>REF[system]]
            series.append({'system':system,'name':name,'version':version,'points':pts})
    figs=HERE/'report_figs'; figs.mkdir(exist_ok=True)
    for system in ('C','N2eq'): svg(system,[s for s in series if s['system']==system],figs/f'logerr_{system}_e5000.svg')
    done=sorted((r for r in s5k if r['status']=='complete'),key=lambda r:(r['system'],float(r['tail500_error_mha'])))
    lines=['# E5000 intermediate completed-only summary','', '| System | Rank | Run | Tail-500 mean ± SEM (Ha) | Error (mHa) | Elapsed | Peak GPU MiB |','|---|---:|---|---:|---:|---:|---:|']
    rank=defaultdict(int)
    for r in done:
        rank[r['system']]+=1; sec=float(r['elapsed_seconds']); elapsed=f'{int(sec//60)}m {int(sec%60)}s'
        lines.append(f"| {r['system']} | {rank[r['system']]} | `{r['run_name']}` | {float(r['tail500_energy_mean']):.9f} ± {float(r['tail500_energy_sem']):.2e} | {float(r['tail500_error_mha']):.3f} | {elapsed} | {float(r['peak_gpu_memory_mib']):.0f} |")
    (HERE/'e5000_intermediate.md').write_text('\n'.join(lines)+'\n')
    print('Wrote completed-only table and two overlay SVGs')
if __name__=='__main__': main()
