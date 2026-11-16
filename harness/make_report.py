"""Assemble the final deliverable: per-video metrics table + README."""
import os, json, csv, numpy as np

SP = os.environ['SP']; OUT = os.environ.get('OUT', os.path.expanduser('~/neuroseg_eval'))
EV = f'{SP}/cv/eval'
FOLD = {3:1,7:1,4:2,5:2,1:3,6:3,2:4,9:4,8:5,10:5}
rows = []
for v in range(1, 11):
    p = f'{EV}/video{v}/metrics.json'
    if os.path.exists(p):
        r = json.load(open(p)); r['video_id'] = v; r['fold'] = FOLD[v]; rows.append(r)
rows.sort(key=lambda r: r['video_id'])
if not rows:
    print('no results'); raise SystemExit

P = np.array([r['precision'] for r in rows]); R = np.array([r['recall'] for r in rows])
F = np.array([r['f1'] for r in rows]); I = np.array([r['mean_iou'] for r in rows])
tp = sum(r['tp'] for r in rows); fp = sum(r['fp'] for r in rows); fn = sum(r['fn'] for r in rows)
pp = tp/(tp+fp) if tp+fp else 0; pr = tp/(tp+fn) if tp+fn else 0
pf = 2*pp*pr/(pp+pr) if pp+pr else 0

os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/metrics_per_video.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['video_id','cv_fold','n_gt','n_predicted','tp','fp','fn','precision','recall','f1','mean_iou_matched'])
    for r in rows:
        w.writerow([r['video_id'],r['fold'],r['n_gt'],r['n_pred'],r['tp'],r['fp'],r['fn'],
                    f"{r['precision']:.4f}",f"{r['recall']:.4f}",f"{r['f1']:.4f}",f"{r['mean_iou']:.4f}"])
    w.writerow(['MEAN','','','','','','',f'{P.mean():.4f}',f'{R.mean():.4f}',f'{F.mean():.4f}',f'{I.mean():.4f}'])
    w.writerow(['SD','','','','','','',f'{P.std():.4f}',f'{R.std():.4f}',f'{F.std():.4f}',f'{I.std():.4f}'])
    w.writerow(['POOLED','',sum(r['n_gt'] for r in rows),sum(r['n_pred'] for r in rows),tp,fp,fn,
                f'{pp:.4f}',f'{pr:.4f}',f'{pf:.4f}',''])

lines = ['| video | fold | N_gt | N_pred | TP | FP | FN | Precision | Recall | F1 | mean IoU |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    lines.append(f"| {r['video_id']} | {r['fold']} | {r['n_gt']} | {r['n_pred']} | {r['tp']} | {r['fp']} | "
                 f"{r['fn']} | {r['precision']:.4f} | {r['recall']:.4f} | **{r['f1']:.4f}** | {r['mean_iou']:.3f} |")
lines.append(f"| **mean** | | | | | | | **{P.mean():.4f}** | **{R.mean():.4f}** | **{F.mean():.4f}** | {I.mean():.3f} |")
lines.append(f"| **SD** | | | | | | | {P.std():.4f} | {R.std():.4f} | {F.std():.4f} | |")
lines.append(f"| **pooled** | | {sum(r['n_gt'] for r in rows)} | {sum(r['n_pred'] for r in rows)} | {tp} | {fp} | {fn} | {pp:.4f} | {pr:.4f} | {pf:.4f} | |")
open(f'{OUT}/_table.md','w').write('\n'.join(lines))
print('\n'.join(lines))
print(f'\nvideos completed: {len(rows)}/10')
