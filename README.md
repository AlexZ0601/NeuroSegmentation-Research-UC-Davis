# UC Davis Neurosegmentation Model

Neuron segmentation research from UC Davis. This repo retrains and evaluates
NeuroSeg-III (Wu et al. 2024, a YOLOv8-seg based model) for finding neurons in
two-photon calcium imaging videos from the Allen Brain Observatory (ABO), and
extracts each neuron's fluorescence trace.

## Repo layout

| path | what |
|---|---|
| `harness/` | dataset building, training, evaluation, trace extraction, reporting |
| `weights/` | the five trained checkpoints (one per cross-validation fold) and the fold 1 training curve |
| `metrics_per_video.csv`, `_table.md` | per-video scores |
| `per_video/video<N>/metrics.json` | per-video scores as JSON |
| `ground_truth/` | traces and stats computed from the ground-truth ROIs |
| `figures/` | mask overlays and trace plots for every video |

The large per-video prediction arrays (`pred_masks.npy`, `pred_F_raw.npy`,
`pred_dff.npy`, `neuroseg_video<N>.mat`, about 900 MB total) are not in git.
Regenerate them with `harness/run_neuroseg_eval.py --save-traces`.

---

## NeuroSeg-III on ABO: trained from scratch, 5-fold cross-validated

The published fine-tuned checkpoint isn't actually available (see
"Provenance"), so the model was **retrained** here and evaluated on all 10 ABO
videos. Every number below comes from a model that never saw that video.

### 1. Per-video results  (`metrics_per_video.csv`)

Standard definitions: **Precision = TP / N_detected**, **Recall = TP / N_GT**.
Instance matching is one-to-one Hungarian assignment at **IoU ≥ 0.5** (paper §2.6, eq. 9).

| video | fold | N_gt | N_pred | TP | FP | FN | Precision | Recall | F1 | mean IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 237 | 239 | 186 | 53 | 51 | 0.7782 | 0.7848 | **0.7815** | 0.784 |
| 2 | 4 | 176 | 141 | 108 | 33 | 68 | 0.7660 | 0.6136 | **0.6814** | 0.779 |
| 3 | 1 | 355 | 309 | 232 | 77 | 123 | 0.7508 | 0.6535 | **0.6988** | 0.762 |
| 4 | 2 | 282 | 244 | 196 | 48 | 86 | 0.8033 | 0.6950 | **0.7452** | 0.762 |
| 5 | 2 | 62 | 79 | 43 | 36 | 19 | 0.5443 | 0.6935 | **0.6099** | 0.683 |
| 6 | 3 | 65 | 99 | 36 | 63 | 29 | 0.3636 | 0.5538 | **0.4390** | 0.659 |
| 7 | 1 | 55 | 46 | 27 | 19 | 28 | 0.5870 | 0.4909 | **0.5347** | 0.693 |
| 8 | 5 | 151 | 107 | 70 | 37 | 81 | 0.6542 | 0.4636 | **0.5426** | 0.655 |
| 9 | 4 | 80 | 83 | 39 | 44 | 41 | 0.4699 | 0.4875 | **0.4785** | 0.634 |
| 10 | 5 | 100 | 68 | 53 | 15 | 47 | 0.7794 | 0.5300 | **0.6310** | 0.702 |
| **mean** | | | | | | | **0.6497** | **0.5966** | **0.6143** | 0.711 |
| **SD** | | | | | | | 0.1449 | 0.1025 | 0.1085 | |
| **pooled** | | 1563 | 1415 | 990 | 425 | 573 | 0.6996 | 0.6334 | 0.6649 | |

**This is below the paper's reported ~0.90 on ABO** (Wu et al. 2024, Fig. 6:
NeuroSeg-III 0.9044; CITE-On 0.885; SUNS 0.881; STNeuroNet 0.8521; CaImAn 0.8035;
Suite2p 0.7616). The gap is real, not a reporting artifact; see §4.

> **Metric-definition warning.** The paper's eq. 5–6 define
> `Precision = N_TP/N_GT` and `Recall = N_TP/N_detected`, which are **swapped**
> relative to standard usage. Their "precision" is what everyone else calls recall.
> F1 is symmetric and therefore comparable; P and R columns are **not**.
> The table above uses the standard definitions.

### 2. Masks and traces  (`per_video/video<N>/`, arrays not in git)

| file | contents |
|---|---|
| `neuroseg_video<N>.mat` | MATLAB: `pred_masks` (487×487×N_pred), `F_raw`, `dff`, `precision`, `recall`, `f1` |
| `pred_masks.npy` | predicted instance masks, bool |
| `pred_F_raw.npy` | area-mean fluorescence per predicted neuron, (N_pred × 5000) |
| `pred_dff.npy` | ΔF/F₀ per predicted neuron (F₀ = 10th percentile of its own trace) |
| `metrics.json` | that video's scores |

`ground_truth/` holds the equivalent traces computed from the **ground-truth** ROIs
for all 10 videos, as a reference set independent of model quality.

`figures/video<N>_masks.png` overlays GT (yellow) vs prediction (cyan);
`figures/video<N>_traces.png` shows ΔF/F for the top 30 predicted neurons by SNR.

### 3. Method (as run)

Following Wu et al. 2024:
- **Input** (§2.2): max projection fused **1:1 with a correlation map** (eq. 1,
  Gaussian-weighted local correlation), normalised. 500-frame windows, stride 250.
- **Model**: YOLOv8s-seg + C2f_Faster_EMA + BiFPN, `nc=1`, 6,452,943 params.
- **Init**: TiCo self-supervised backbone (see "Upstream bugs": the released
  loader transfers nothing; fixed here).
- **Training** (§2.5.2): 200 epochs, batch 8, mosaic disabled for the final 10,
  early stopping. Apple M5 Pro / MPS.
- **Protocol**: 5-fold cross-validation by video, 8 train / 2 held-out, equivalent
  to the paper's best-reported **split=8**. Folds balanced by neuron count:
  `[3,7] [4,5] [1,6] [2,9] [8,10]`.
- **Labels**: a neuron is labelled in a window only if it is detectable there
  (peak ΔF/F ≥ 0.3). Labelling silent cells in every window teaches hallucination.
  Verified: active cells have median ROI/background contrast 1.70 vs 1.09 inactive.

**Deviation from the paper:** it uses 20 videos (10 at 175 µm to train, 10 at
275 µm to validate) and therefore measures *cross-depth* generalisation. Only 10
videos (one depth) were available here, so this is within-depth cross-validation.
It is not the same generalisation test.

### 4. Why the gap: measured, not guessed

Diagnostics on video 3 (the sweep script is not included in this repo):

- **91.5%** of GT neurons are visible (contrast > 1.05) in *some* window.
- **76.9%** is the model's detection ceiling: the fraction matched by *any* raw
  detection across all 19 windows at conf 0.05, before merging.
- **65.4%** is final merged recall.

So ~9% of GT is effectively invisible (silent cells), ~15 points is genuine model
limitation, and the merge step costs ~11 points.

**Confidence tuning is not the lever.** A full sweep of conf × min-votes moves F1
only between 0.66 and 0.71 (best 0.705 at conf 0.30, vs 0.699 at the default 0.25).

**Longer windows are worse, not better**. Tested, contrary to expectation:

| correlation window | GT visible (>1.05), video 3 | video 7 |
|---|---|---|
| 500 frames (used) | **0.915** | **0.982** |
| 2500 frames | 0.800 | 0.873 |
| full video | 0.656 | 0.818 |

**The dominant effect is neuron density.** F1 correlates strongly with N_gt:
Pearson **r = +0.764 (p = 0.010)**, Spearman ρ = +0.794 (p = 0.006).

| | F1 | Precision | Recall | N_pred/N_gt |
|---|---|---|---|---|
| dense (N_gt ≥ 150, n=5) | 0.690 | 0.751 | 0.642 | 0.851 |
| sparse (N_gt < 150, n=5) | 0.539 | 0.549 | 0.551 | **1.070** |

Sparse videos fail by **over-predicting** (video 6: 99 predicted vs 65 actual).
The training pool is dominated by dense videos (237–355 neurons), so the model
appears to have learned a density prior it then imposes on sparse recordings.
Mean matched IoU stays 0.63–0.78 throughout: when a neuron is found it is
segmented well, so the loss is in **detection, not delineation**.

### 5. Upstream bugs found and fixed

1. **The SSL backbone never loaded.** `SSL_Finetuning.py:load_backbone()` calls
   `load_state_dict(ckpt, strict=False)`, but checkpoint keys are `backbone.N.*`
   while model keys are `model.N.*`, so **0 of 340 match**, silently. As published,
   the self-supervised stage contributes nothing to fine-tuning. Fixed by remapping
   the prefix (90 tensors / 3,149,039 params, backbone layers 0–9). This affects how
   the paper's TiCo ablation (Fig. 4C) should be read.
2. **`np.trapz`** (removed in NumPy 2.0) in `ultralytics/utils/metrics.py:407`.
   Crashes AP computation, but only once the model gets its first true positives.
3. **uint8 overflow** in `polygons2masks_overlap`. Instance indices are stored in
   uint8 and NumPy 2.0 (NEP 50) no longer upcasts `uint8_array * python_int`, so any
   image with >255 instances raises `OverflowError`. With ~100 neurons/image and
   mosaic stitching 4 together, this triggers constantly on dense data.
4. `torch.load` needs `weights_only=False` on PyTorch ≥ 2.6 (`harness/torch_compat.py`).

### 6. Provenance of the missing checkpoint

Neither supplied source contains a fine-tuned neuron model:
- The `NeuroSeg3` zip ships stock **COCO** `yolov8s-seg.pt` (classes `person`,
  `bicycle`, …) and a **backbone-only** TiCo checkpoint.
- `github.com/zimo-5kk/NeuroSeg3` is byte-identical (blob SHAs `b6226914…`,
  `e07e37ce…`), has no `runs/`, and its extra ONNX export is the same COCO model:
  output tensor `[1, 116, 8400]` = 4 box + **80 classes** + 32 mask coeffs, whereas
  a 1-class neuron model would be `[1, 37, 8400]`.
- The Drive folder holds 34 `LSTM_Data_*.mat` files only.

### 7. Reproducing

```bash
python harness/build_dataset.py --videos <dir> --rois <dir> --gt-traces <dir> \
    --out ds --split pool --ids 1 2 3 4 5 6 7 8 9 10 --win 500 --stride 250
python harness/train_fold.py --pool ds/pool --work cv --fold 1 \
    --train-ids 1 2 4 5 6 8 9 10 --val-ids 3 7 --epochs 200
python harness/run_neuroseg_eval.py --weights cv/runs/fold1/weights/best.pt \
    --videos <dir> --rois <dir> --out cv/eval --ids 3 7 --save-traces
```
Set `NEUROSEG_REPO` to the NeuroSeg3 checkout. `weights/neuroseg3_fold<N>.pt` are
the five trained checkpoints; each is valid only for its own held-out videos.
