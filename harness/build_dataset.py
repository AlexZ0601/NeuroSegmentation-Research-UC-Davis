"""Build a YOLO-seg dataset from ABO videos + ground-truth ROI masks.

Each sample is a max projection over a window of frames. A neuron is labelled in
that window only if it is actually detectable there (peak dF/F within the window
above a threshold) -- labelling every ROI in every window would train the model to
hallucinate silent cells.
"""
import os, argparse
import numpy as np, scipy.io as sio, tifffile, cv2
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from fusion import fused_image


def norm8(x):
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.9)
    if hi <= lo: hi = lo + 1
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def mask_to_polys(m, min_area=15, eps_frac=0.004):
    """Binary mask -> list of normalised YOLO-seg polygons."""
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = m.shape; out = []
    for c in cs:
        if cv2.contourArea(c) < min_area or len(c) < 3:
            continue
        c = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)
        if len(c) < 3:
            continue
        p = c.reshape(-1, 2).astype(np.float32)
        p[:, 0] /= W; p[:, 1] /= H
        out.append(np.clip(p, 0, 1))
    return out


def build_video(vid, tiff, roi, dff, out_img, out_lbl, win, stride, act_thr, prefix=''):
    seg = sio.loadmat(roi)['seg'].astype(bool)
    mov = tifffile.imread(tiff)                      # (T,H,W) uint8, ~1.2 GB
    T, H, W = mov.shape
    seg = seg[:H, :W, :]; N = seg.shape[2]
    polys = [mask_to_polys(seg[:, :, k]) for k in range(N)]   # contours once, not per window
    n_written = 0; labels_per_img = []
    for z in range(0, T - win + 1, stride):
        img = fused_image(mov[z:z + win])   # paper Sec 2.2: maxproj + correlation map, 1:1
        active = np.nonzero(dff[:, z:z + win].max(1) >= act_thr)[0]
        lines = ['0 ' + ' '.join(f'{v:.6f}' for v in p.reshape(-1))
                 for k in active for p in polys[k]]
        if not lines:
            continue
        name = f'{prefix}v{vid}_z{z:05d}'
        cv2.imwrite(os.path.join(out_img, name + '.png'), img)
        open(os.path.join(out_lbl, name + '.txt'), 'w').write('\n'.join(lines) + '\n')
        n_written += 1; labels_per_img.append(len(lines))
    del mov
    return n_written, N, float(np.mean(labels_per_img)) if labels_per_img else 0.0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True); ap.add_argument('--rois', required=True)
    ap.add_argument('--gt-traces', required=True, help='dir with video<N>/dff.npy')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ids', nargs='+', type=int, required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--win', type=int, default=200); ap.add_argument('--stride', type=int, default=100)
    ap.add_argument('--act-thr', type=float, default=0.5)
    a = ap.parse_args()

    oi = os.path.join(a.out, a.split, 'images'); ol = os.path.join(a.out, a.split, 'labels')
    os.makedirs(oi, exist_ok=True); os.makedirs(ol, exist_ok=True)
    for vid in a.ids:
        dff = np.load(os.path.join(a.gt_traces, f'video{vid}', 'dff.npy'))
        n, N, avg = build_video(vid, os.path.join(a.videos, f'{vid}.tiff'),
                                os.path.join(a.rois, f'{vid}_ROI.mat'), dff, oi, ol,
                                a.win, a.stride, a.act_thr)
        print(f'video {vid}: {n} images, {N} GT neurons, {avg:.1f} labels/image', flush=True)
