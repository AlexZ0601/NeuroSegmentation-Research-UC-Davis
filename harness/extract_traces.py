"""Extract per-neuron fluorescence traces from an ABO video using ground-truth ROI masks.

Usage: extract_traces.py <video_id> <tiff_path> <roi_mat_path> <out_dir>
Writes:  F_raw.npy, dff.npy, traces.csv, neuron_stats.csv, traces.mat, maxproj.npy
"""
import sys, os, json
import numpy as np, scipy.io as sio, scipy.sparse as sp, tifffile

vid, tiff_path, roi_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(out_dir, exist_ok=True)

seg = sio.loadmat(roi_path)['seg'].astype(bool)           # (H0, W0, N)
with tifffile.TiffFile(tiff_path) as tf:
    T = len(tf.pages); H, W = tf.pages[0].shape
    seg = seg[:H, :W, :]                                   # top-left crop (verified alignment)
    N = seg.shape[2]
    flat = seg.reshape(-1, N).T                            # (N, H*W)
    areas = flat.sum(1).astype(np.int64)
    M = sp.csr_matrix(flat.astype(np.float32))
    Mn = sp.diags(1.0 / areas.astype(np.float32)) @ M      # area-normalised -> mean per ROI

    F = np.zeros((N, T), np.float32)
    maxproj = np.zeros((H, W), np.uint8)
    meanacc = np.zeros((H, W), np.float64)
    CH = 250
    for s in range(0, T, CH):
        e = min(s + CH, T)
        blk = np.stack([tf.pages[i].asarray() for i in range(s, e)]).astype(np.float32)
        maxproj = np.maximum(maxproj, blk.max(0).astype(np.uint8))
        meanacc += blk.sum(0)
        F[:, s:e] = Mn @ blk.reshape(e - s, -1).T
        print(f'  video {vid}: frames {e}/{T}', flush=True)
    meanimg = (meanacc / T)

# dF/F with F0 = 10th percentile of each neuron's own trace (standard baseline for ABO)
F0 = np.percentile(F, 10, axis=1, keepdims=True)
F0 = np.maximum(F0, 1e-3)
dff = (F - F0) / F0

# per-neuron summary stats
ys, xs = [], []
for k in range(N):
    yy, xx = np.nonzero(seg[:, :, k]); ys.append(yy.mean()); xs.append(xx.mean())
noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / 0.9539   # robust sigma
snr = dff.max(1) / np.maximum(noise, 1e-6)

np.save(f'{out_dir}/F_raw.npy', F)
np.save(f'{out_dir}/dff.npy', dff)
np.save(f'{out_dir}/maxproj.npy', maxproj)
np.save(f'{out_dir}/meanimg.npy', meanimg.astype(np.float32))
np.save(f'{out_dir}/masks.npy', seg)
sio.savemat(f'{out_dir}/traces_video{vid}.mat',
            {'F_raw': F, 'dff': dff, 'area': areas, 'centroid_y': np.array(ys),
             'centroid_x': np.array(xs), 'snr': snr, 'maxproj': maxproj})

import csv
with open(f'{out_dir}/neuron_stats.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['video_id','neuron_id','area_px','centroid_y','centroid_x',
                                   'F0','mean_dff','peak_dff','noise_sigma','snr'])
    for k in range(N):
        w.writerow([vid, k+1, areas[k], f'{ys[k]:.2f}', f'{xs[k]:.2f}', f'{F0[k,0]:.4f}',
                    f'{dff[k].mean():.4f}', f'{dff[k].max():.4f}', f'{noise[k]:.4f}', f'{snr[k]:.2f}'])

np.savetxt(f'{out_dir}/traces_dff.csv', dff, delimiter=',', fmt='%.5f')
json.dump({'video_id': vid, 'n_neurons': int(N), 'n_frames': int(T), 'height': int(H),
           'width': int(W), 'source': 'ground-truth ROI masks (not model output)'},
          open(f'{out_dir}/meta.json','w'), indent=2)
print(f'DONE video {vid}: N={N} T={T}')
