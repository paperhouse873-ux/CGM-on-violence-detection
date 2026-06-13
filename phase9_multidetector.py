"""
Phase 9 — Evaluate CGM across many detectors + headroom curve
=========================================================================
Attach CGM (962 params, runs on CPU) to each detector's p_base, measure with the
STRICT protocol used in Phase 7:
  * ORIGINAL split (cache/splits.npy), scaler fit on train only, select by val AUC,
    many seeds -> mean±std.
  * report: FPR@matched-recall + AUC + Δα for the WHOLE test and the HARD group.

Auto-detect detectors that already have p_base:
  x3d_s          -> cache/p_base.npy
  slowfast_r50   -> cache/p_base_slowfast_r50.npy
  mvit_base_16x4 -> cache/p_base_mvit_base_16x4.npy

Context streams (crowd/light/motion) are SHARED across all detectors.

HEADROOM curve: each (detector, dataset) is one point
  x = baseline weakness (1 - AUC_baseline)
  y = CGM benefit       (FPR@matched-recall reduction)
-> expectation: the weaker the baseline, the more CGM helps.
The RLVS point (cross-domain) is read from results/phase6b_indomain_results.json if present.

Output:
  results/phase9_multidetector.json
  figures/RQ_multidetector_table.png
  figures/RQ_headroom_curve.png

Run:  python phase9_multidetector.py --seeds 10
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE = Path("cache")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
FIGURES = Path("figures"); FIGURES.mkdir(exist_ok=True)
HARD_LO, HARD_HI = 0.3, 0.7

DETECTORS = [
    ("X3D-S",    CACHE / "p_base.npy"),
    ("SlowFast", CACHE / "p_base_slowfast_r50.npy"),
    ("MViT",     CACHE / "p_base_mvit_base_16x4.npy"),
]


class ContextGatingModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(32, 1), nn.Sigmoid())
        self.ctx = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, pb):
        a = self.gate(x).squeeze(1)
        pc = self.ctx(x).squeeze(1)
        return a * pb + (1 - a) * pc, a, pc


def tpr_at(y, p, thr=0.5):
    pr = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) else 0.0


def fpr_at_recall(y, p, t):
    fpr, tpr, _ = roc_curve(y, p)
    ok = tpr >= t - 1e-9
    return float(fpr[np.argmax(ok)]) if ok.any() else None


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(set(y.tolist())) > 1 else float("nan")


def train_infer(p_base, ctx, y, splits, seed):
    tr, va = splits == 0, splits == 1
    X = np.concatenate([p_base.reshape(-1, 1), ctx], axis=1)
    sc = StandardScaler().fit(X[tr]); Xn = sc.transform(X).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    m = ContextGatingModule(X.shape[1]); crit = nn.BCELoss()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    t = lambda a: torch.tensor(a)
    Xtr, ytr, pbtr = t(Xn[tr]), t(y[tr].astype(np.float32)), t(p_base[tr].astype(np.float32))
    Xva, pbva = t(Xn[va]), t(p_base[va].astype(np.float32))
    best, bs, wait = -1, None, 0
    for _ in range(300):
        m.train(); pf, _, _ = m(Xtr, pbtr); loss = crit(pf, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pfv, _, _ = m(Xva, pbva)
        va_auc = safe_auc(y[va], pfv.numpy()); sch.step(va_auc)
        if va_auc > best:
            best, wait = va_auc, 0
            bs = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
        if wait >= 30:
            break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad():
        pf, a, _ = m(t(Xn), t(p_base.astype(np.float32)))
    return pf.numpy(), a.numpy()


def grp(y, pb, pf, mask):
    ys, pbs, pfs = y[mask], pb[mask], pf[mask]
    if len(ys) == 0 or len(set(ys.tolist())) < 2:
        return None
    t0 = tpr_at(ys, pbs)
    return dict(mr_b=fpr_at_recall(ys, pbs, t0), mr_a=fpr_at_recall(ys, pfs, t0),
                auc_b=safe_auc(ys, pbs), auc_a=safe_auc(ys, pfs))


def eval_detector(name, p_base, ctx, y, splits, seeds):
    te = splits == 2
    hard = te & (p_base > HARD_LO) & (p_base < HARD_HI)
    ov, hd, das = [], [], []
    for s in range(seeds):
        pf, a = train_infer(p_base, ctx, y, splits, s)
        ov.append(grp(y, p_base, pf, te))
        hd.append(grp(y, p_base, pf, hard))
        yte = y[te]
        das.append(abs(a[te][yte == 1].mean() - a[te][yte == 0].mean()))

    def ag(lst, k):
        v = [d[k] for d in lst if d and d[k] is not None and not np.isnan(d[k])]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), 0.0)

    return dict(
        name=name, n_test=int(te.sum()), n_hard=int(hard.sum()),
        overall=dict(mr_b=ag(ov, "mr_b"), mr_a=ag(ov, "mr_a"),
                     auc_b=ag(ov, "auc_b"), auc_a=ag(ov, "auc_a")),
        hard=dict(mr_b=ag(hd, "mr_b"), mr_a=ag(hd, "mr_a"),
                  auc_b=ag(hd, "auc_b"), auc_a=ag(hd, "auc_a")),
        d_alpha=(float(np.mean(das)), float(np.std(das))),
    )


def main(args):
    y = np.load(CACHE / "labels.npy").astype(int)
    splits = np.load(CACHE / "splits.npy")
    ctx = np.concatenate([np.load(CACHE / "z_crowd.npy"),
                          np.load(CACHE / "z_light.npy"),
                          np.load(CACHE / "z_motion.npy")], axis=1)

    print("=" * 72)
    print(f"PHASE 9 — CGM across detectors (RWF, original split, {args.seeds} seeds)")
    print("=" * 72)

    rows, headroom = [], []
    for name, path in DETECTORS:
        if not path.exists():
            print(f"\n  [SKIP] {name}: no {path.name} yet (run phase8b after fine-tuning).")
            continue
        p_base = np.load(path)
        if len(p_base) != len(y):
            print(f"\n  [ERROR] {name}: p_base length mismatch ({len(p_base)} vs {len(y)}). Skip.")
            continue
        r = eval_detector(name, p_base, ctx, y, splits, args.seeds)
        rows.append(r)
        ob, oa = r["overall"]["mr_b"][0], r["overall"]["mr_a"][0]
        hb, ha = r["hard"]["mr_b"][0], r["hard"]["mr_a"][0]
        aub, aua = r["overall"]["auc_b"][0], r["overall"]["auc_a"][0]
        print(f"\n  ── {name} (test={r['n_test']}, hard={r['n_hard']}) ──")
        print(f"     OVERALL  FPR@mr {ob:.4f}->{oa:.4f} (Δ={ob-oa:+.4f}) | "
              f"AUC {aub:.4f}->{aua:.4f}")
        print(f"     HARD     FPR@mr {hb:.4f}->{ha:.4f} (Δ={hb-ha:+.4f}) | "
              f"AUC {r['hard']['auc_b'][0]:.4f}->{r['hard']['auc_a'][0]:.4f}")
        print(f"     |Δα|={r['d_alpha'][0]:.4f}")
        headroom.append((name, "RWF", 1 - aub, ob - oa))

    # add the RLVS (cross-domain) point if available
    p6b = RESULTS / "phase6b_indomain_results.json"
    if p6b.exists():
        d = json.load(open(p6b))
        aub = d["baseline"].get("auc"); mrb = d["baseline"].get("fpr_mr")
        mra = d["in_domain_cgm"].get("fpr_mr")
        if aub and mrb is not None and mra is not None:
            headroom.append(("X3D-S", "RLVS(cross)", 1 - aub, mrb - mra))
            print(f"\n  + RLVS cross-domain point: weakness={1-aub:.3f}, benefit={mrb-mra:+.4f}")

    # save JSON
    def native(o):
        if isinstance(o, dict): return {k: native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [native(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return o
    json.dump(native(dict(seeds=args.seeds, detectors=rows, headroom=headroom)),
              open(RESULTS / "phase9_multidetector.json", "w"), indent=2)
    print(f"\n  saved -> results/phase9_multidetector.json")

    # ── FIGURE: headroom curve ──
    if len(headroom) >= 2:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        xs = [h[2] for h in headroom]; ys = [h[3] for h in headroom]
        ax.scatter(xs, ys, s=70, color="#2e6fb7", zorder=3)
        for nm, ds, x, yv in headroom:
            ax.annotate(f"{nm}\n{ds}", (x, yv), textcoords="offset points",
                        xytext=(6, 6), fontsize=8)
        if len(xs) >= 2:
            z = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 50)
            ax.plot(xx, np.polyval(z, xx), "--", color="gray",
                    label=f"trend (slope={z[0]:.2f})")
            ax.legend(frameon=False)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("Baseline weakness  (1 − AUC)")
        ax.set_ylabel("CGM benefit  (FPR@matched-recall reduction)")
        ax.set_title("Headroom law: weaker detector → larger CGM benefit")
        fig.tight_layout(); fig.savefig(FIGURES / "RQ_headroom_curve.png", dpi=160)
        print(f"  saved -> figures/RQ_headroom_curve.png")
    else:
        print("  (need >=2 points to draw the headroom curve — rerun after SlowFast/MViT)")

    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    main(args)
