"""Train NeuroSeg-III (YOLOv8s-seg + C2f_Faster_EMA + BiFPN, nc=1) for one CV fold,
initialised from the TiCo self-supervised backbone.

NOTE: the upstream repo's FinetuneYolo.load_backbone() does
      model.load_state_dict(ckpt, strict=False) with a checkpoint whose keys are
      'backbone.N.*' while the model's are 'model.N.*' -- 0 of 340 keys match, so
      upstream silently trains from scratch. We remap the prefix so the SSL
      weights actually transfer (90 tensors / 3.15M params, backbone layers 0-9).
"""
import os, sys, argparse, shutil, yaml, contextlib, io

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch_compat                                   # noqa: F401
REPO = os.environ.get('NEUROSEG_REPO')
sys.path.insert(0, REPO)
import torch
from ultralytics import YOLO


def load_tico_backbone(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu')
    msd = model.model.state_dict()
    rem = {'model.' + k[len('backbone.'):]: v for k, v in ck.items() if k.startswith('backbone.')}
    ok = {k: v for k, v in rem.items() if k in msd and msd[k].shape == v.shape}
    msd.update(ok)
    model.model.load_state_dict(msd)
    n = sum(v.numel() for v in ok.values())
    print(f'[TiCo] transferred {len(ok)} tensors / {n:,} params', flush=True)
    return model


def link_split(pool, ids, dst):
    for sub in ('images', 'labels'):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
    n = 0
    for f in sorted(os.listdir(os.path.join(pool, 'images'))):
        vid = int(f.split('_')[0][1:])
        if vid not in ids:
            continue
        stem = os.path.splitext(f)[0]
        for sub, ext in (('images', '.png'), ('labels', '.txt')):
            s = os.path.join(pool, sub, stem + ext)
            d = os.path.join(dst, sub, stem + ext)
            if not os.path.exists(d):
                os.symlink(os.path.abspath(s), d)
        n += 1
    return n


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', required=True); ap.add_argument('--work', required=True)
    ap.add_argument('--fold', required=True)
    ap.add_argument('--train-ids', nargs='+', type=int, required=True)
    ap.add_argument('--val-ids', nargs='+', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=150); ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=640); ap.add_argument('--device', default='mps')
    a = ap.parse_args()

    root = os.path.join(a.work, f'fold{a.fold}')
    shutil.rmtree(root, ignore_errors=True); os.makedirs(root, exist_ok=True)
    ntr = link_split(a.pool, a.train_ids, os.path.join(root, 'train'))
    nva = link_split(a.pool, a.val_ids, os.path.join(root, 'val'))
    print(f'fold {a.fold}: train {ntr} imgs (videos {a.train_ids}) | val {nva} imgs (videos {a.val_ids})', flush=True)

    cfg = os.path.join(root, 'data.yaml')
    yaml.dump({'path': os.path.abspath(root), 'train': 'train/images',
               'val': 'val/images', 'names': {0: 'neuron'}}, open(cfg, 'w'))

    m = YOLO(os.path.join(REPO, 'ultralytics/cfg/models/v8/yolov8s-seg-C2f-Faster-EMA-bifpn.yaml'))
    load_tico_backbone(m, os.path.join(REPO, 'weights/ssl/TiCo/yolov8s-seg-TiCo-best.pt'))

    m.train(data=cfg, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
            project=os.path.join(a.work, 'runs'), name=f'fold{a.fold}',
            amp=False, close_mosaic=10, patience=60, workers=4, cache=True,
            degrees=180.0, fliplr=0.5, flipud=0.5, scale=0.3, translate=0.1,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, mosaic=1.0, plots=False, val=True, exist_ok=True)
