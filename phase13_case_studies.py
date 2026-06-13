"""
Phase 13 — Qualitative case studies (Task 4)
============================================
Show what the CGM "looks at" via 3 concrete cases from the HARD group of the test set:
  (A) False alarm RESCUED by CGM : nonFight, p_base>=0.5 -> p_cgm<0.5
  (B) True positive KEPT         : fight,   both base and CGM correct
  (C) Failure case (honest)      : a case CGM handles wrong (kept wrong, or broke it)

Use the deployed CGM (checkpoints/cgm_e4.pth + cgm_scaler.pkl) so the numbers match
the "product" model. Map cache index -> path by split.json order (train+val+test),
same as the phase3/phase8b pipeline.

Output:
  figures/FIG7_case_studies.png
  results/phase13_cases.json

Run:  python phase13_case_studies.py --root RWF-2000
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE = Path("cache")
CKPT = Path("checkpoints")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
FIGURES = Path("figures"); FIGURES.mkdir(exist_ok=True)
HARD_LO, HARD_HI = 0.3, 0.7


class ContextGatingModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(32, 1), nn.Sigmoid())
        self.ctx = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, pb):
        a = self.gate(x).squeeze(1)
        c = self.ctx(x).squeeze(1)
        return a * pb + (1 - a) * c, a, c


def load_paths_in_cache_order():
    """split.json: train+val+test in the exact cache order (phase3/phase8b)."""
    sj = json.load(open("split.json", encoding="utf-8"))
    paths = []
    for key in ("train", "val", "test"):
        for item in sj[key]:
            paths.append(item["path"])
    return paths


def grab_frames(video_path, k=4):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        # read sequentially
        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()
        if not frames:
            return [np.zeros((224, 224, 3), np.uint8)] * k
        idxs = np.linspace(0, len(frames) - 1, k).astype(int)
        return [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idxs]
    idxs = np.linspace(0, n - 1, k).astype(int)
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, f = cap.read()
        if not ret:
            f = np.zeros((224, 224, 3), np.uint8)
        else:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        out.append(f)
    cap.release()
    return out


def main(args):
    root = Path(args.root)
    p_base = np.load(CACHE / "p_base.npy")
    z_crowd = np.load(CACHE / "z_crowd.npy")
    z_light = np.load(CACHE / "z_light.npy")
    z_motion = np.load(CACHE / "z_motion.npy")
    y = np.load(CACHE / "labels.npy").astype(int)
    splits = np.load(CACHE / "splits.npy")

    X = np.concatenate([p_base.reshape(-1, 1), z_crowd, z_light, z_motion], axis=1)
    scaler = pickle.load(open(CKPT / "cgm_scaler.pkl", "rb"))
    Xn = scaler.transform(X).astype(np.float32)

    ck = torch.load(CKPT / "cgm_e4.pth", map_location="cpu", weights_only=False)
    model = ContextGatingModule(13)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    with torch.no_grad():
        pf, alpha, pctx = model(torch.tensor(Xn), torch.tensor(p_base.astype(np.float32)))
    pf = pf.numpy(); alpha = alpha.numpy()

    paths = load_paths_in_cache_order()
    assert len(paths) == len(p_base), f"path/cache mismatch {len(paths)} vs {len(p_base)}"

    te = splits == 2
    hard = te & (p_base > HARD_LO) & (p_base < HARD_HI)
    idx_all = np.arange(len(p_base))

    # ── pick 3 cases ────────────────────────────────────────────────────────
    def pick(mask, sort_key, reverse=True):
        cand = idx_all[mask]
        if len(cand) == 0:
            return None
        cand = sorted(cand, key=lambda i: sort_key(i), reverse=reverse)
        return cand[0]

    # (A) rescued FP: nonFight, base wrong (>=0.5), CGM correct (<0.5) — prefer biggest drop
    mask_A = hard & (y == 0) & (p_base >= 0.5) & (pf < 0.5)
    iA = pick(mask_A, lambda i: p_base[i] - pf[i])

    # (B) preserved TP: fight, base correct (>=0.5) and CGM correct (>=0.5)
    mask_B = hard & (y == 1) & (p_base >= 0.5) & (pf >= 0.5)
    iB = pick(mask_B, lambda i: pf[i])

    # (C) failure: prefer a case CGM BREAKS (fight, base correct but CGM wrong),
    #     otherwise take a nonFight still kept as FP
    mask_C1 = hard & (y == 1) & (p_base >= 0.5) & (pf < 0.5)   # CGM breaks a TP
    mask_C2 = hard & (y == 0) & (p_base >= 0.5) & (pf >= 0.5)  # FP still remains
    iC = pick(mask_C1, lambda i: p_base[i] - pf[i])
    fail_kind = "CGM lowered a correct violent prediction below 0.5"
    if iC is None:
        iC = pick(mask_C2, lambda i: pf[i])
        fail_kind = "Persisting false alarm: context did not rescue it"

    cases = []
    for tag, i, desc in [
        ("A: Rescued false alarm", iA, "Non-violent clip; CGM pulls p below 0.5"),
        ("B: Preserved true positive", iB, "Violent clip; CGM keeps p above 0.5"),
        ("C: Failure case", iC, fail_kind),
    ]:
        if i is None:
            print(f"  [WARN] no candidate for {tag}")
            continue
        cases.append(dict(tag=tag, idx=int(i), path=paths[i],
                          label=int(y[i]), p_base=float(p_base[i]),
                          p_cgm=float(pf[i]), alpha=float(alpha[i]), desc=desc))
        print(f"  {tag}: {paths[i]} | y={y[i]} p_base={p_base[i]:.3f} "
              f"-> p_cgm={pf[i]:.3f} (α={alpha[i]:.2f})")

    # ── figure: 3 cases as COLUMNS, 2 frames each (first–last) → compact strip ──
    K = 2  # frames per case (first, last)
    ncol = len(cases)
    fig, axes = plt.subplots(K, ncol, figsize=(2.7 * ncol, 2.5 * K))
    if ncol == 1:
        axes = axes[:, None]
    LBL = {0: "non-violent", 1: "violent"}
    for c_i, c in enumerate(cases):
        frames = grab_frames(root / c["path"], K)
        verdict_b = "FIGHT" if c["p_base"] >= 0.5 else "SAFE"
        verdict_c = "FIGHT" if c["p_cgm"] >= 0.5 else "SAFE"
        for r in range(K):
            ax = axes[r, c_i]
            ax.imshow(frames[r]); ax.set_xticks([]); ax.set_yticks([])
        head = (f"{c['tag']}\nGT: {LBL[c['label']]}\n"
                f"base {c['p_base']:.2f}[{verdict_b}] $\\to$ "
                f"CGM {c['p_cgm']:.2f}[{verdict_c}]")
        axes[0, c_i].set_title(head, fontsize=8.0)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.02,
                        hspace=0.05, wspace=0.05)
    fig.savefig(FIGURES / "FIG7_case_studies.png", dpi=160, bbox_inches="tight")
    print(f"\n  saved -> figures/FIG7_case_studies.png")

    json.dump(cases, open(RESULTS / "phase13_cases.json", "w"), indent=2)
    print(f"  saved -> results/phase13_cases.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="RWF-2000")
    args = ap.parse_args()
    main(args)
