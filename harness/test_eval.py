"""Sanity checks for the matching metric. Usage: python harness/test_eval.py <path/to/5_ROI.mat>
(any ABO ground-truth ROI file with a 'seg' array works)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_masks import evaluate
import scipy.io as sio

gt = sio.loadmat(sys.argv[1])['seg'].astype(bool)
print('GT shape', gt.shape)

# 1. identity -> perfect
r = evaluate(gt, gt); print(f"identity          P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} tp={r['tp']} fp={r['fp']} fn={r['fn']}")
assert r['f1'] == 1.0

# 2. drop 20% of GT -> recall 0.8, precision 1.0
keep = gt[:, :, :50]
r = evaluate(gt, keep); print(f"drop 12/62        P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} tp={r['tp']} fp={r['fp']} fn={r['fn']}")

# 3. add 10 spurious blobs -> precision drops
extra = np.zeros((488,488,10), bool)
for i in range(10): extra[10+i*3:16+i*3, 400:406, i] = True
r = evaluate(gt, np.concatenate([gt, extra], 2)); print(f"+10 false pos     P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} tp={r['tp']} fp={r['fp']} fn={r['fn']}")

# 4. shift every mask by 3px -> partial IoU degradation
sh = np.roll(gt, 3, axis=0)
r = evaluate(gt, sh); print(f"3px shift         P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} meanIoU={r['mean_iou']:.3f}")
sh = np.roll(gt, 8, axis=0)
r = evaluate(gt, sh); print(f"8px shift         P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} meanIoU={r['mean_iou']:.3f}")
print('metric core OK')
