"""Instance-level Precision / Recall / F1 for neuron segmentation.

Matches predicted instance masks to ground-truth ROIs one-to-one by IoU
(Hungarian assignment, IoU >= thr counts as a true positive) -- the standard
protocol used for ABO / Neurofinder neuron segmentation.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment


def _to_csr(masks):
    """(H, W, N) bool array -> (N, H*W) sparse."""
    H, W, N = masks.shape
    return sp.csr_matrix(masks.reshape(-1, N).T.astype(np.float32))


def iou_matrix(gt, pred):
    """gt: (H,W,Ng) bool, pred: (H,W,Np) bool -> (Ng, Np) IoU."""
    G, P = _to_csr(gt), _to_csr(pred)
    inter = np.asarray((G @ P.T).todense())
    ag = np.asarray(G.sum(1)).ravel()[:, None]
    ap = np.asarray(P.sum(1)).ravel()[None, :]
    union = ag + ap - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def evaluate(gt, pred, thr=0.5):
    """Returns dict with precision, recall, f1, tp, fp, fn, mean IoU of matches."""
    Ng, Np = gt.shape[2], pred.shape[2]
    if Ng == 0 or Np == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=Np, fn=Ng,
                    mean_iou=0.0, n_gt=Ng, n_pred=Np, matches=[])
    M = iou_matrix(gt, pred)
    r, c = linear_sum_assignment(-M)
    keep = M[r, c] >= thr
    r, c, ious = r[keep], c[keep], M[r, c][keep]
    tp = int(keep.sum()); fp = Np - tp; fn = Ng - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn,
                mean_iou=float(ious.mean()) if tp else 0.0,
                n_gt=Ng, n_pred=Np,
                matches=list(zip((r + 1).tolist(), (c + 1).tolist(), ious.tolist())))
