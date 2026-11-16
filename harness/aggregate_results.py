"""Collect per-video CV results into the final table + per-video figures."""
import os, json, sys, csv
import numpy as np, scipy.io as sio, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SP = os.environ['SP']
EV = f'{SP}/cv/eval'; GT = f'{SP}/out/gt'; OUT = sys.argv[1] if len(sys.argv) > 1 else f'{SP}/final'
os.makedirs(f'{OUT}/figures', exist_ok=True)

FOLD = {3: 1, 7: 1, 4: 2, 5: 2, 1: 3, 6: 3, 2: 4, 9: 4, 8: 5, 10: 5}
rows = []
for v in range(1, 11):
    p = f'{EV}/video{v}/metrics.json'
    if not os.path.exists(p):
        continue
    r = json.load(open(p)); r['video_id'] = v; r['fold'] = FOLD[v]
    rows.append(r)

if not rows:
    print('no results yet'); sys.exit(0)

rows.sort(key=lambda r: r['video_id'])
hdr = ['video_id', 'fold', 'n_gt', 'n_pred', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'mean_iou']
with open(f'{OUT}/metrics_per_video.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(hdr)
    for r in rows:
        w.writerow([r['video_id'], r['fold'], r['n_gt'], r['n_pred'], r['tp'], r['fp'], r['fn'],
                    f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1']:.4f}", f"{r['mean_iou']:.4f}"])
    P = [r['precision'] for r in rows]; R = [r['recall'] for r in rows]; F = [r['f1'] for r in rows]
    tp = sum(r['tp'] for r in rows); fp = sum(r['fp'] for r in rows); fn = sum(r['fn'] for r in rows)
    mp = tp / (tp + fp) if tp + fp else 0; mr = tp / (tp + fn) if tp + fn else 0
    w.writerow(['MEAN', '', sum(r['n_gt'] for r in rows), sum(r['n_pred'] for r in rows), tp, fp, fn,
                f'{np.mean(P):.4f}', f'{np.mean(R):.4f}', f'{np.mean(F):.4f}',
                f"{np.mean([r['mean_iou'] for r in rows]):.4f}"])
    w.writerow(['POOLED', '', '', '', tp, fp, fn, f'{mp:.4f}', f'{mr:.4f}',
                f'{2*mp*mr/(mp+mr) if mp+mr else 0:.4f}', ''])

print(f"{'vid':>4} {'fold':>4} {'n_gt':>5} {'n_pred':>6} {'tp':>4} {'fp':>4} {'fn':>4} "
      f"{'P':>7} {'R':>7} {'F1':>7} {'mIoU':>6}")
for r in rows:
    print(f"{r['video_id']:>4} {r['fold']:>4} {r['n_gt']:>5} {r['n_pred']:>6} {r['tp']:>4} {r['fp']:>4} "
          f"{r['fn']:>4} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} {r['mean_iou']:>6.3f}")
print(f"{'MEAN':>4} {'':>4} {'':>5} {'':>6} {'':>4} {'':>4} {'':>4} "
      f"{np.mean(P):>7.4f} {np.mean(R):>7.4f} {np.mean(F):>7.4f}")
print(f"{'POOL':>4} {'':>4} {'':>5} {'':>6} {tp:>4} {fp:>4} {fn:>4} {mp:>7.4f} {mr:>7.4f} "
      f"{2*mp*mr/(mp+mr) if mp+mr else 0:>7.4f}")

# per-video figure: GT vs predicted masks + predicted traces
for r in rows:
    v = r['video_id']
    pred = np.load(f'{EV}/video{v}/pred_masks.npy')
    mp_img = np.load(f'{GT}/video{v}/maxproj.npy')
    gtm = sio.loadmat(f'{SP}/data/rois/{v}_ROI.mat')['seg'].astype(bool)[:487, :487, :]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.8))
    ax[0].imshow(mp_img, cmap='gray'); ax[0].set_title(f'video {v} — max projection')
    ax[1].imshow(mp_img, cmap='gray')
    for k in range(gtm.shape[2]):
        ax[1].contour(gtm[:, :, k], [0.5], colors=['#ffd400'], linewidths=0.4)
    ax[1].set_title(f'ground truth (N={gtm.shape[2]})')
    ax[2].imshow(mp_img, cmap='gray')
    for k in range(pred.shape[2]):
        ax[2].contour(pred[:, :, k], [0.5], colors=['#00e5ff'], linewidths=0.4)
    ax[2].set_title(f"NeuroSeg-III (N={pred.shape[2]}) — F1={r['f1']:.3f}")
    for a in ax: a.axis('off')
    plt.tight_layout(); plt.savefig(f'{OUT}/figures/video{v}_masks.png', dpi=130, bbox_inches='tight')
    plt.close()

    dp = f'{EV}/video{v}/pred_dff.npy'
    if os.path.exists(dp):
        d = np.load(dp)
        if d.shape[0]:
            noise = np.median(np.abs(np.diff(d, 1)), 1) / 0.9539
            snr = d.max(1) / np.maximum(noise, 1e-6)
            o = np.argsort(-snr)[:30]
            fig, a = plt.subplots(figsize=(13, 8))
            t = np.arange(d.shape[1]) / 30.0
            for i, k in enumerate(o):
                tr = d[k]; a.plot(t, tr / max(tr.max(), 1e-6) * 0.9 + i, lw=0.45,
                                  color=plt.cm.viridis(i / max(len(o), 1)))
            a.set_xlabel('time (s, 30 Hz)'); a.set_ylabel('predicted neuron (top 30 by SNR)')
            a.set_title(f'video {v} — ΔF/F from NeuroSeg-III predicted masks')
            a.margins(x=0.005); plt.tight_layout()
            plt.savefig(f'{OUT}/figures/video{v}_traces.png', dpi=130); plt.close()
print('\nwrote', OUT)
