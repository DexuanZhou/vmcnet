#!/usr/bin/env python3
"""Reporting-only SVG figures for the completed N2 E50000 variance audit."""
import csv,html,math,statistics,subprocess
from pathlib import Path
import numpy as np

EXP=Path('/scratch/dexuan1/vmcnet/experiments/N2_adcomp_E50000_compare')
AUD=EXP/'results/variance_audit'; REPO=Path('/scratch/dexuan1/report_repo'); FIG=REPO/'report_figs'; FIG.mkdir(exist_ok=True)
ROOT=Path('/scratch/dexuan1/runs/N2_adcomp_E50000_compare')
METHODS=[('A_rank800_hard','rank800 hard','#0072B2',''),('B_rank400_beta02','rank400 adaptive beta=0.2','#D55E00','6,4')]

def readcsv(p): return list(csv.DictReader(open(p)))
data={k:readcsv(ROOT/k/'training_metrics.csv')[-5000:] for k,_,_,_ in METHODS}
blocks=readcsv(AUD/'final5000_blocks.csv'); audit=readcsv(AUD/'summary.csv')
def esc(x):return html.escape(str(x))
class SVG:
 def __init__(self,w=1200,h=720):self.w=w;self.h=h;self.e=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><style>text{{font-family:Arial,sans-serif;fill:#222}} .axis{{stroke:#333;stroke-width:1}} .grid{{stroke:#ddd;stroke-width:1}} </style>']
 def text(self,x,y,s,size=14,anchor='middle',weight='normal',rotate=None):self.e.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}"'+(f' transform="rotate({rotate} {x} {y})"' if rotate else '')+f'>{esc(s)}</text>')
 def line(self,x1,y1,x2,y2,color='#333',width=1,dash=''):self.e.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>' )
 def rect(self,x,y,w,h,fill='none',stroke='none',opacity=1):self.e.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" opacity="{opacity}"/>')
 def circle(self,x,y,r,fill,stroke='white'):self.e.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" stroke="{stroke}"/>')
 def poly(self,pts,color,width=1.5,dash=''):self.e.append(f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="'+ ' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)+'"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>' )
 def save(self,p):self.e.append('</svg>');p.write_text('\n'.join(self.e))

def panel(s,x0,y0,w,h,xs,ys,xlabel,ylabel,title,logy=False,yr=None,xticks=5,yticks=5):
 tx=lambda x:x0+(x-min(xs))/(max(xs)-min(xs) or 1)*w
 vals=[math.log10(max(y,1e-12)) for y in ys] if logy else ys
 lo,hi=(min(vals),max(vals)) if yr is None else ((math.log10(yr[0]),math.log10(yr[1])) if logy else yr)
 if hi==lo:hi=lo+1
 pad=.04*(hi-lo);lo-=pad;hi+=pad
 ty=lambda y:y0+h-(math.log10(max(y,1e-12)) if logy else y-lo)/(hi-lo)*h if logy else y0+h-(y-lo)/(hi-lo)*h
 # Correct log transform mapping separately for readability.
 if logy: ty=lambda y:y0+h-(math.log10(max(y,1e-12))-lo)/(hi-lo)*h
 s.line(x0,y0+h,x0+w,y0+h);s.line(x0,y0,x0,y0+h)
 for i in range(yticks+1):
  v=lo+(hi-lo)*i/yticks;y=y0+h-h*i/yticks;s.line(x0,y,x0+w,y,'#e6e6e6');lab=f'{10**v:.2g}' if logy else f'{v:.3g}';s.text(x0-8,y+5,lab,11,'end')
 for i in range(xticks+1):
  v=min(xs)+(max(xs)-min(xs))*i/xticks;x=tx(v);s.line(x,y0+h,x,y0+h+5);s.text(x,y0+h+20,f'{v:.0f}',11)
 s.text(x0+w/2,y0+h+43,xlabel,13);s.text(x0-58,y0+h/2,ylabel,13,rotate=-90);s.text(x0+w/2,y0-12,title,16,weight='bold')
 return tx,ty,(lo,hi)

def legend(s,y=35):
 x=780
 for _,label,c,d in METHODS:s.line(x,y,x+35,y,c,3,d);s.text(x+43,y+5,label,13,'start');x+=210

def convert(svg):
 png=svg.with_suffix('.png');subprocess.run(['rsvg-convert','-w','1800','-o',str(png),str(svg)],check=True)

# 1 linear and log time series.
for log,suffix in [(False,''),(True,'_log')]:
 s=SVG();s.text(600,28,'N2 final-5000 per-epoch raw variance'+(' (log scale)' if log else ''),21,weight='bold');legend(s,58)
 allv=[float(r['variance_noclip']) for k,_,_,_ in METHODS for r in data[k]];xs=list(range(45001,50001));tx,ty,_=panel(s,90,95,1050,550,xs,allv,'Epoch','Raw variance (Ha²)','',logy=log)
 for a,b,c,label in [(47001,47100,'#0072B2','rank800 burst'),(45201,45300,'#D55E00','rank400 burst'),(47401,47500,'#D55E00','rank400 burst')]:s.rect(tx(a),95,tx(b)-tx(a),550,c,'none',.10);s.text((tx(a)+tx(b))/2,115,label,10)
 for k,_,c,d in METHODS:s.poly([(tx(int(r['epoch'])),ty(float(r['variance_noclip']))) for r in data[k]],c,1,d)
 p=FIG/f'n2_final5000_raw_variance_timeseries{suffix}.svg';s.save(p);convert(p)

# 2 block means.
s=SVG();s.text(600,28,'N2 final-5000 non-overlapping 100-epoch block means',21,weight='bold');legend(s,58);xs=list(range(1,51));ys=[float(r['raw_variance_mean']) for r in blocks];tx,ty,_=panel(s,90,95,1050,550,xs,ys,'Block index','Block mean raw variance (Ha²)','')
for k,_,c,d in METHODS:
 z=[r for r in blocks if r['run']==k];v=[float(r['raw_variance_mean']) for r in z];med=statistics.median(v);s.line(tx(1),ty(med),tx(50),ty(med),c,1.5,'3,3');s.poly([(tx(i+1),ty(x)) for i,x in enumerate(v)],c,2,d);im=int(np.argmax(v));s.circle(tx(im+1),ty(v[im]),6,c);s.text(tx(im+1),ty(v[im])-10,f'max {v[im]:.2f}',11)
p=FIG/'n2_final5000_block_means.svg';s.save(p);convert(p)

# 3 histogram and ECDF in log-x space.
s=SVG(1300,650);s.text(650,28,'Distribution of per-epoch raw variance over final 5000 epochs',21,weight='bold');legend(s,58)
vals={k:np.array([float(r['variance_noclip']) for r in data[k]]) for k,_,_,_ in METHODS};alllog=np.concatenate([np.log10(x) for x in vals.values()]);bins=np.linspace(alllog.min(),alllog.max(),55)
for pi,title in enumerate(['Histogram (log variance bins)','ECDF (log x-axis)']):
 x0=75+pi*625;y0=105;w=535;h=430;s.line(x0,y0+h,x0+w,y0+h);s.line(x0,y0,x0,y0+h);s.text(x0+w/2,y0-12,title,16,weight='bold');s.text(x0+w/2,y0+h+38,'Raw variance (Ha²)',13)
 tx=lambda z:x0+(z-bins[0])/(bins[-1]-bins[0])*w
 for i in range(6):z=bins[0]+(bins[-1]-bins[0])*i/5;s.text(tx(z),y0+h+19,f'{10**z:.2g}',11)
 for k,_,c,d in METHODS:
  if pi==0:
   hist,_=np.histogram(np.log10(vals[k]),bins=bins,density=True);mx=max(np.histogram(np.log10(x),bins=bins,density=True)[0].max() for x in vals.values());pts=[]
   for j,v in enumerate(hist):pts.extend([(tx(bins[j]),y0+h-v/mx*h),(tx(bins[j+1]),y0+h-v/mx*h)])
  else:
   z=np.sort(np.log10(vals[k]));pts=[(tx(x),y0+h-(i+1)/len(z)*h) for i,x in enumerate(z)]
  s.poly(pts,c,2,d)
for j,(k,label,c,d) in enumerate(METHODS):
 v=vals[k];txt=f'{label}: mean {v.mean():.3f}, median {np.median(v):.3f}, p99 {np.quantile(v,.99):.2f}, p99.9 {np.quantile(v,.999):.2f}, max {v.max():.1f}';s.text(75,585+j*21,txt,12,'start')
p=FIG/'n2_final5000_variance_distribution.svg';s.save(p);convert(p)

# 4 raw versus clipped, one panel per method.
s=SVG(1300,780);s.text(650,28,'Raw and clipped variance over final 5000 epochs',21,weight='bold')
for pi,(k,label,c,d) in enumerate(METHODS):
 z=data[k];xs=[int(r['epoch']) for r in z];raw=[float(r['variance_noclip']) for r in z];clip=[float(r['variance']) for r in z];tx,ty,_=panel(s,90,85+pi*335,1130,250,xs,raw+clip,'Epoch','Variance (Ha²)',label,logy=True);s.poly([(tx(x),ty(y)) for x,y in zip(xs,raw)],c,1,d);s.poly([(tx(x),ty(y)) for x,y in zip(xs,clip)],'#333',1.5,'3,3');s.text(980,105+pi*335,'raw',12,'start');s.line(940,101+pi*335,972,101+pi*335,c,2,d);s.text(1080,105+pi*335,'clipped',12,'start');s.line(1035,101+pi*335,1072,101+pi*335,'#333',2,'3,3')
p=FIG/'n2_final5000_raw_vs_clipped.svg';s.save(p);convert(p)

s=SVG();s.text(600,28,'Raw/clipped variance ratio over final 5000 epochs',21,weight='bold');legend(s,58);xs=list(range(45001,50001));rat=[]
for k,_,_,_ in METHODS:rat += [float(r['variance_noclip'])/float(r['variance']) for r in data[k]]
tx,ty,_=panel(s,90,95,1050,550,xs,rat,'Epoch','Raw/clipped variance ratio','')
for k,_,c,d in METHODS:s.poly([(tx(int(r['epoch'])),ty(float(r['variance_noclip'])/float(r['variance']))) for r in data[k]],c,1,d)
p=FIG/'n2_final5000_raw_clipped_ratio.svg';s.save(p);convert(p)

# 5 eight-metric small-multiple bars.
meta={};
for k,_,_,_ in METHODS:
 q=readcsv(Path(str(ROOT/k)+'.metadata')/'phase_timing.csv');t=[float(x['monotonic_seconds']) for x in q if x['event']=='epoch_end'];gp=readcsv(Path(str(ROOT/k)+'.metadata')/'gpu_memory_poll.csv');meta[k]=(statistics.median([b-a for a,b in zip(t,t[1:])][10:]),max(float(x['memory_used_mib']) for x in gp))
sm={(r['run'],int(r['window'])):r for r in audit};metrics=[('Final-1000 mean raw var',lambda k:float(sm[k,1000]['mean_per_epoch_raw_variance']),'Ha²'),('Final-5000 mean raw var',lambda k:float(sm[k,5000]['mean_per_epoch_raw_variance']),'Ha²'),('Final-5000 median raw var',lambda k:float(sm[k,5000]['median_per_epoch_raw_variance']),'Ha²'),('Final-5000 maximum raw var',lambda k:float(sm[k,5000]['variance_max']),'Ha²'),('Final-5000 block median',lambda k:statistics.median(float(x['raw_variance_mean']) for x in blocks if x['run']==k),'Ha²'),('Maximum block mean',lambda k:max(float(x['raw_variance_mean']) for x in blocks if x['run']==k),'Ha²'),('Seconds per epoch',lambda k:meta[k][0],'s'),('Peak GPU memory',lambda k:meta[k][1]/1024,'GiB')]
s=SVG(1400,760);s.text(700,28,'N2 E50000 variance and cost summary',22,weight='bold')
for idx,(title,fn,unit) in enumerate(metrics):
 col=idx%4;row=idx//4;x0=45+col*345;y0=70+row*340;w=285;h=245;vv=[fn(k) for k,_,_,_ in METHODS];mx=max(vv)*1.13
 s.text(x0+w/2,y0,title,14,weight='bold');s.line(x0,y0+h,x0+w,y0+h)
 for j,((k,label,c,d),v) in enumerate(zip(METHODS,vv)):
  bw=70;x=x0+45+j*135;bh=v/mx*(h-45);s.rect(x,y0+h-bh,bw,bh,c);s.text(x+bw/2,y0+h-bh-7,f'{v:.3g}',12);s.text(x+bw/2,y0+h+18,'rank800' if j==0 else 'rank400 β=.2',10)
 s.text(x0+5,y0+22,unit,11,'start')
p=FIG/'n2_final5000_summary_comparison.svg';s.save(p);convert(p)

# 6 raw energy and rolling100.
s=SVG();s.text(600,28,'N2 final-5000 raw energy stability',21,weight='bold');legend(s,58);xs=list(range(45001,50001));ys=[float(r['energy_noclip']) for k,_,_,_ in METHODS for r in data[k]];tx,ty,_=panel(s,90,95,1050,550,xs,ys,'Epoch','Raw epoch mean energy (Ha)','')
for k,_,c,d in METHODS:
 v=np.array([float(r['energy_noclip']) for r in data[k]]);roll=np.convolve(v,np.ones(100)/100,mode='valid');s.poly([(tx(x),ty(y)) for x,y in zip(xs,v)],c,.6,d);s.poly([(tx(x),ty(y)) for x,y in zip(xs[99:],roll)],c,3,d)
p=FIG/'n2_final5000_energy_stability.svg';s.save(p);convert(p)

files=sorted(p.name for p in FIG.glob('n2_final5000_*') if p.suffix in ('.svg','.png'))
(REPO/'n2_variance_audit_manifest.txt').write_text('\n'.join(files+['n2_variance_audit_summary.md'])+'\n')
print('\n'.join(str(FIG/f) for f in files))
