"""Per-image evaluation, matching the repo's Segment_test.py protocol (and the
paper's Fig. 6, n=20 images): each fused window image is scored against the
neurons annotated in THAT image, then averaged. Reported alongside the
video-level merged score, which is the harder task of recovering every neuron.
"""
import sys, os, numpy as np, tifffile, scipy.io as sio, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import torch_compat                                    # noqa: F401
sys.path.insert(0, os.environ['NEUROSEG_REPO'])
from fusion import fused_image
from eval_masks import evaluate
from run_neuroseg_eval import predict_instances


def per_image(model, vid, videos, rois, gt_dff, win=500, stride=250,
              conf=0.25, nms_iou=0.5, imgsz=640, max_det=400, act_thr=0.3, iou_thr=0.5):
    mov = tifffile.imread(os.path.join(videos, f'{vid}.tiff'))
    T, H, W = mov.shape
    seg = sio.loadmat(os.path.join(rois, f'{vid}_ROI.mat'))['seg'].astype(bool)[:H, :W, :]
    dff = np.load(gt_dff)
    out = []
    for z in range(0, T - win + 1, stride):
        act = np.nonzero(dff[:, z:z + win].max(1) >= act_thr)[0]
        if len(act) == 0:
            continue
        g = seg[:, :, act]
        p, _ = predict_instances(model, fused_image(mov[z:z + win]), conf, nms_iou, imgsz, max_det)
        out.append(evaluate(g, p, iou_thr))
    return out


if __name__ == '__main__':
    from ultralytics import YOLO
    SP = os.environ['SP']
    w, dev = sys.argv[1], sys.argv[2]
    ids = [int(x) for x in sys.argv[3:]]
    m = YOLO(w); m.to(dev)
    print(f"{'vid':>4} {'imgs':>5} {'P':>8} {'R':>8} {'F1':>8}   (per-image, paper protocol)")
    allf = []
    for v in ids:
        rs = per_image(m, v, f'{SP}/data/videos', f'{SP}/data/rois', f'{SP}/out/gt/video{v}/dff.npy')
        P = np.mean([r['precision'] for r in rs]); R = np.mean([r['recall'] for r in rs])
        F = np.mean([r['f1'] for r in rs]); allf += [r['f1'] for r in rs]
        print(f'{v:>4} {len(rs):>5} {P:>8.4f} {R:>8.4f} {F:>8.4f}')
    print(f'\nmean over images: F1 = {np.mean(allf):.4f} +/- {np.std(allf):.4f}')
