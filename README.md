# Context Gating Module (CGM) for False-Alarm Reduction in Violence Detection

Code and pre-computed features for our paper *"Context-Aware False Alarm
Reduction for Violence Detection in Surveillance Videos Using a Lightweight
Gating Module."*

The idea is simple: instead of building yet another heavier backbone, we keep
an existing violence detector **frozen** and bolt a tiny module on top of it
that decides *how much to trust the detector on each clip* using cheap,
annotation-free scene context (crowd density, lighting, motion). The whole
module is 962 parameters and trains in a few seconds on CPU.

The point of the paper is not a new accuracy record. It is a careful look at
*when* this kind of context gating actually helps. Short answer: mostly on the
hard, ambiguous clips, and mostly when the base detector is weak or used on a
new domain.

## What's in here

```
phase1_dataset.py            # build the RWF-2000 split, sanity checks
phase2_finetune_x3ds.py      # fine-tune X3D-S as the base detector
phase3_extract_context.py    # crowd / lighting / motion features (12-dim)
phase4_train_cgm.py          # train the gating module on top of p_base
phase6b_train_rlvs.py        # cross-dataset: zero-shot vs in-domain on RLVS
phase7_hard_case_analysis.py # easy vs hard split, matched-recall FPR, AUC
phase8_finetune_detectors.py # SlowFast-R50 and MViT-B baselines
phase8b_extract_pbase.py     # cache p_base for the extra detectors
phase9_multidetector.py      # the headroom trend across detectors
phase10_paper_figures.py     # regenerate the figures used in the paper
phase13_case_studies.py      # qualitative examples (rescued / preserved / miss)
phase14_cgm_arch_ablation.py # width / depth / linear / no-gate ablation
phase15_latency.py           # CPU timing for the gate and the context front-end
phase0_step3_statistics.py   # dataset statistics
split.py                     # stratified 70/15/15 split helper
cache/                       # pre-computed p_base and context features
checkpoints/                 # trained CGM weights (cgm_e4.pth) + scaler
results/                     # metrics dumped by the phases (json/csv)
figures/                     # a few key figures from the paper
```

The big backbone checkpoints (X3D-S, SlowFast, MViT) are not included because
they are too large for a Git repo. You can reproduce them with `phase2` and
`phase8`, or just use the cached `p_base` in `cache/` to retrain the gate.

## Quick start

```bash
pip install -r requirements.txt

# The gate is cheap. With the cached features you can retrain it on CPU:
python phase4_train_cgm.py

# Reproduce the headroom analysis and the paper figures:
python phase9_multidetector.py --seeds 10
python phase10_paper_figures.py --seeds 10
```

If you want to start from raw video instead, you need RWF-2000 and RLVS, then
run the phases in order (`phase1` → `phase3` → `phase4`).

## The module

The CGM reads a 13-dim vector (the detector's probability `p_base` plus a
12-dim context descriptor) and produces two things: a trust weight `alpha` and
a context-corrected probability `p_ctx`. The final score is

```
p_final = alpha * p_base + (1 - alpha) * p_ctx
```

`alpha -> 1` means "trust the detector", `alpha -> 0` means "trust the
context". Both branches are just `FC(13->32) -> ReLU -> Dropout -> FC(32->1)`,
962 parameters total.

## Data

- **RWF-2000** — main benchmark, stratified 70/15/15 split (seed 42).
- **RLVS** — used only for the cross-dataset transfer experiment.

Neither dataset is redistributed here; grab them from their original sources
and point the scripts at the folders.

## Notes

- All metrics in `results/` are means over 10 seeds where applicable.
- Context features are detector-independent, so they are extracted once and
  shared across X3D-S / SlowFast / MViT.
- `cache/rlvs/` holds the RLVS features used by `phase6b`.

## Citation
