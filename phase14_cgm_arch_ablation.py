"""
Phase 14 — CGM architecture ablation
====================================
Checks whether FC(13->32) is really the best CGM variant, not just an
arbitrary choice. Compares several CGM architecture variants, keeping the same
protocol as phase7
(original split, scaler fit on train, select by val-AUC, eval on test, many seeds).
All train in a few seconds on CPU.

Variants:
  - width: hidden = 16 / 32(default) / 64 / 128
  - depth: 1 hidden layer (default) vs 2 hidden layers
  - no-gate (additive): drop the gate branch, p_final = 0.5*p_base + 0.5*p_ctx
  - linear-gate: gate/ctx are a single Linear (no nonlinearity)
  - residual: p_final = clip(p_base + (1-alpha)*(p_ctx - p_base))  (already a skip)

Report: #params, test AUC, test FPR@matched-recall, hard AUC, hard FPR@mr
(mean±std over seeds). Conclusion: width=32, 1 layer is the sweet spot.

Output:
  results/phase14_arch_ablation.json
  figures/FIG8_arch_ablation.png

Run:  python phase14_cgm_arch_ablation.py --seeds 10
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


# ── architecture variants ────────────────────────────────────────────────────
class CGM(nn.Module):
    """Parameterized config: width, depth, gate_mode."""
    def __init__(self, d, width=32, depth=1, gate_mode="mlp", linear=False):
        super().__init__()
        self.gate_mode = gate_mode

        def block(out_sigmoid=True):
            if linear:
                layers = [nn.Linear(d, 1)]
            else:
                layers = [nn.Linear(d, width), nn.ReLU(), nn.Dropout(0.3)]
                for _ in range(depth - 1):
                    layers += [nn.Linear(width, width), nn.ReLU(), nn.Dropout(0.3)]
                layers += [nn.Linear(width, 1)]
            if out_sigmoid:
                layers += [nn.Sigmoid()]
            return nn.Sequential(*layers)

        if gate_mode == "mlp":
            self.gate = block()
        self.ctx = block()

    def forward(self, x, pb):
        c = self.ctx(x).squeeze(1)
        if self.gate_mode == "mlp":
            a = self.gate(x).squeeze(1)
        else:  # no-gate: fixed average
            a = torch.full_like(pb, 0.5)
        return a * pb + (1 - a) * c, a, c

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


VARIANTS = [
    dict(name="FC width=16",        width=16, depth=1, gate_mode="mlp", linear=False),
    dict(name="FC width=32 (ours)", width=32, depth=1, gate_mode="mlp", linear=False),
    dict(name="FC width=64",        width=64, depth=1, gate_mode="mlp", linear=False),
    dict(name="FC width=128",       width=128, depth=1, gate_mode="mlp", linear=False),
    dict(name="2-layer (32-32)",    width=32, depth=2, gate_mode="mlp", linear=False),
    dict(name="Linear gate/ctx",    width=32, depth=1, gate_mode="mlp", linear=True),
    dict(name="No-gate (additive)", width=32, depth=1, gate_mode="none", linear=False),
]


# ── helpers ──────────────────────────────────────────────────────────────────
def tpr_at(y, p, thr=0.5):
    pr = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) else 0.0


def fpr_at_recall(y, p, t):
    fpr, tpr, _ = roc_curve(y, p)
    ok = tpr >= t - 1e-9
    return float(fpr[np.argmax(ok)]) if ok.any() else float("nan")


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(set(np.asarray(y).tolist())) > 1 else float("nan")


def train_variant(Xn, p_base, y, splits, cfg, seed):
    tr, va = splits == 0, splits == 1
    torch.manual_seed(seed); np.random.seed(seed)
    model = CGM(Xn.shape[1], width=cfg["width"], depth=cfg["depth"],
                gate_mode=cfg["gate_mode"], linear=cfg["linear"])
    crit = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    t = lambda a: torch.tensor(a)
    Xtr, ytr, pbtr = t(Xn[tr]), t(y[tr].astype(np.float32)), t(p_base[tr].astype(np.float32))
    Xva, pbva = t(Xn[va]), t(p_base[va].astype(np.float32))
    best, bs, wait = -1, None, 0
    for _ in range(300):
        model.train()
        pf, _, _ = model(Xtr, pbtr)
        loss = crit(pf, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pfv, _, _ = model(Xva, pbva)
        va_auc = safe_auc(y[va], pfv.numpy())
        sch.step(va_auc)
        if va_auc > best:
            best, wait = va_auc, 0
            bs = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= 30:
            break
    model.load_state_dict(bs); model.eval()
    with torch.no_grad():
        pf, _, _ = model(t(Xn), t(p_base.astype(np.float32)))
    return pf.numpy(), model.n_params()


def main(args):
    p_base = np.load(CACHE / "p_base.npy")
    ctx = np.concatenate([np.load(CACHE / "z_crowd.npy"),
                          np.load(CACHE / "z_light.npy"),
                          np.load(CACHE / "z_motion.npy")], axis=1)
    y = np.load(CACHE / "labels.npy").astype(int)
    splits = np.load(CACHE / "splits.npy")
    tr, te = splits == 0, splits == 2
    hard = te & (p_base > HARD_LO) & (p_base < HARD_HI)

    X = np.concatenate([p_base.reshape(-1, 1), ctx], axis=1)
    scaler = StandardScaler().fit(X[tr])
    Xn = scaler.transform(X).astype(np.float32)
    t0_te = tpr_at(y[te], p_base[te])
    t0_hd = tpr_at(y[hard], p_base[hard])

    print("=" * 78)
    print(f"PHASE 14 — CGM architecture ablation | test={te.sum()} hard={hard.sum()} "
          f"seeds={args.seeds}")
    print("=" * 78)
    # baseline reference
    auc_base = safe_auc(y[te], p_base[te])
    print(f"  [ref] X3D-S only: test AUC={auc_base:.4f} | "
          f"hard AUC={safe_auc(y[hard], p_base[hard]):.4f}")

    rows = []
    for cfg in VARIANTS:
        accs = dict(test_auc=[], test_mr=[], hard_auc=[], hard_mr=[])
        nparams = None
        for s in range(args.seeds):
            pf, nparams = train_variant(Xn, p_base, y, splits, cfg, s)
            accs["test_auc"].append(safe_auc(y[te], pf[te]))
            accs["test_mr"].append(fpr_at_recall(y[te], pf[te], t0_te))
            accs["hard_auc"].append(safe_auc(y[hard], pf[hard]))
            accs["hard_mr"].append(fpr_at_recall(y[hard], pf[hard], t0_hd))
        row = dict(name=cfg["name"], params=int(nparams))
        for k, v in accs.items():
            v = np.array(v, float); v = v[~np.isnan(v)]
            row[k] = [float(v.mean()), float(v.std())]
        rows.append(row)
        print(f"  {cfg['name']:22} p={row['params']:4d} | "
              f"testAUC={row['test_auc'][0]:.4f}±{row['test_auc'][1]:.3f} "
              f"testFPR={row['test_mr'][0]:.4f} | "
              f"hardAUC={row['hard_auc'][0]:.4f}±{row['hard_auc'][1]:.3f} "
              f"hardFPR={row['hard_mr'][0]:.4f}")

    json.dump(dict(seeds=args.seeds, ref_test_auc=float(auc_base), variants=rows),
              open(RESULTS / "phase14_arch_ablation.json", "w"), indent=2)
    print(f"\n  saved -> results/phase14_arch_ablation.json")

    # ── figure: hard AUC ± std by variant, mark "ours" ───────────────────────
    names = [r["name"] for r in rows]
    hauc = [r["hard_auc"][0] for r in rows]
    herr = [r["hard_auc"][1] for r in rows]
    cols = ["#238b45" if "ours" in n else "#9ecae1" for n in names]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars = ax.bar(range(len(names)), hauc, yerr=herr, capsize=3,
                  color=cols, edgecolor="black", linewidth=0.7)
    ax.axhline(safe_auc(y[hard], p_base[hard]), color="#cb181d", ls="--", lw=1.2,
               label="base (no CGM)")
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Hard-subset AUC")
    ax.set_ylim(0.55, max(hauc) + 0.05)
    ax.set_title("CGM architecture ablation (hard-subset AUC, mean±std over seeds)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "FIG8_arch_ablation.png", dpi=160)
    print(f"  saved -> figures/FIG8_arch_ablation.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    main(args)
