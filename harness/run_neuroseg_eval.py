#!/usr/bin/env python
"""NeuroSeg-III evaluation on ABO videos.

tiff -> max projections over WIN frames (same preprocessing as training)
     -> YOLOv8-seg inference per window
     -> instances merged across windows (score-ordered NMS)
     -> one-to-one IoU matching vs ground-truth ROI  -> P / R / F1
     -> per-neuron dF/F traces from the PREDICTED masks.
"""
import argparse, os, sys, json, csv
import numpy as np, scipy.io as sio, scipy.sparse as sp, tifffile, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch_compat                                   # noqa: F401
sys.path.insert(0, os.environ.get('NEUROSEG_REPO', os.path.join(HERE, '..', 'work', 'NeuroSeg3')))
from eval_masks import evaluate                       # noqa: E402
from fusion import fused_image                        # noqa: E402


def norm8(x):
    """Must match build_dataset.norm8 exactly."""
    lo, hi = np.percentile(x, 0.5), np.percentile(x, 99.9)
    if hi <= lo: hi = lo + 1
    return np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def predict_instances(model, img, conf, iou, imgsz, max_det):
    H, W = img.shape
    r = model.predict(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), imgsz=imgsz, conf=conf,
                      iou=iou, max_det=max_det, retina_masks=True, verbose=False)[0]
    if r.masks is None or len(r.masks) == 0:
        return np.zeros((H, W, 0), bool), np.zeros(0, np.float32)
    m = r.masks.data.cpu().numpy()
    if m.shape[1:] != (H, W):
        m = np.stack([cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST) for x in m])
    return (m > 0.5).transpose(1, 2, 0), r.boxes.conf.cpu().numpy()


def merge_instances(masks, scores, iou_thr, min_votes, min_area=20):
    """Fuse the same neuron seen in several windows; keep those seen >= min_votes times."""
    if masks.shape[2] == 0:
        return masks
    H, W, N = masks.shape
    flat = sp.csr_matrix(masks.reshape(-1, N).T.astype(np.float32))
    area = np.asarray(flat.sum(1)).ravel()
    cand = np.nonzero(area >= min_area)[0]
    cand = cand[np.argsort(-scores[cand])]
    kept, votes = [], []
    for i in cand:
        if kept:
            inter = np.asarray((flat[i] @ flat[kept].T).todense()).ravel()
            union = area[i] + area[kept] - inter
            ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            j = int(np.argmax(ious))
            if ious[j] > iou_thr:
                votes[j] += 1
                continue
        kept.append(i); votes.append(1)
    sel = [k for k, v in zip(kept, votes) if v >= min_votes]
    return masks[:, :, sel] if sel else np.zeros((H, W, 0), bool)


def traces_from_masks(mov, masks, chunk=500):
    if masks.shape[2] == 0:
        return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
    T = mov.shape[0]; N = masks.shape[2]
    M = sp.csr_matrix(masks.reshape(-1, N).T.astype(np.float32))
    areas = np.asarray(M.sum(1)).ravel()
    Mn = sp.diags(1.0 / np.maximum(areas, 1)) @ M
    F = np.zeros((N, T), np.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        F[:, s:e] = Mn @ mov[s:e].reshape(e - s, -1).T.astype(np.float32)
    F0 = np.maximum(np.percentile(F, 10, axis=1, keepdims=True), 1e-3)
    return F, (F - F0) / F0


def run_video(model, vid, videos, rois, a):
    tif = os.path.join(videos, f'{vid}.tiff')
    mov = tifffile.imread(tif)
    T, H, W = mov.shape
    gt = sio.loadmat(os.path.join(rois, f'{vid}_ROI.mat'))['seg'].astype(bool)[:H, :W, :]
    ms, ss = [], []
    for z in range(0, T - a.win + 1, a.stride):
        img = fused_image(mov[z:z + a.win])   # must match build_dataset
        m, s = predict_instances(model, img, a.conf, a.nms_iou, a.imgsz, a.max_det)
        if m.shape[2]:
            ms.append(m); ss.append(s)
    pred = (merge_instances(np.concatenate(ms, 2), np.concatenate(ss), a.merge_iou, a.min_votes)
            if ms else np.zeros((H, W, 0), bool))
    r = evaluate(gt, pred, a.iou_thr); r['video_id'] = vid
    return r, pred, mov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--videos', required=True); ap.add_argument('--rois', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--ids', nargs='+', type=int, required=True)
    ap.add_argument('--win', type=int, default=500); ap.add_argument('--stride', type=int, default=250)
    ap.add_argument('--iou-thr', type=float, default=0.5); ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--nms-iou', type=float, default=0.5); ap.add_argument('--merge-iou', type=float, default=0.5)
    ap.add_argument('--min-votes', type=int, default=1); ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--max-det', type=int, default=300); ap.add_argument('--device', default='mps')
    ap.add_argument('--save-traces', action='store_true')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights); model.to(a.device)
    os.makedirs(a.out, exist_ok=True)
    for vid in a.ids:
        r, pred, mov = run_video(model, vid, a.videos, a.rois, a)
        print(f"video {vid}: N_gt={r['n_gt']} N_pred={r['n_pred']} tp={r['tp']} fp={r['fp']} fn={r['fn']} "
              f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} mIoU={r['mean_iou']:.4f}", flush=True)
        vd = os.path.join(a.out, f'video{vid}'); os.makedirs(vd, exist_ok=True)
        json.dump({k: v for k, v in r.items() if k != 'matches'}, open(f'{vd}/metrics.json', 'w'), indent=2)
        np.save(f'{vd}/pred_masks.npy', pred)
        if a.save_traces:
            F, dff = traces_from_masks(mov, pred)
            np.save(f'{vd}/pred_F_raw.npy', F); np.save(f'{vd}/pred_dff.npy', dff)
            sio.savemat(f'{vd}/neuroseg_video{vid}.mat',
                        {'pred_masks': pred, 'F_raw': F, 'dff': dff,
                         'precision': r['precision'], 'recall': r['recall'], 'f1': r['f1']})
        del mov


if __name__ == '__main__':
    main()
