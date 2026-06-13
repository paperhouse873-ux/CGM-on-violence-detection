"""
Phase 7 (Direction 1) — "hard case" analysis: where does CGM help on RWF?
================================================================
Claim: X3D-S is already very good on EASY cases. The real contribution of CGM
is on the HARD group — where the detector is unsure (p_base near 0.5) — by
injecting context signal (crowd/light/motion) that a pure visual detector
doesn't see.

STRICT protocol (survive review):
  * Use the ORIGINAL split (cache/splits.npy): train CGM on train, select by val
    AUC, evaluate on TEST -> test p_base does NOT leak (X3D-S never saw test).
  * Scaler fit on train ONLY.
  * Many seeds -> mean ± std.
  * Report: matched-recall FPR + AUC + Δα for the WHOLE test, and SPLIT by
    EASY vs HARD group (by p_base uncertainty).
  * Plus: context-only strength (AUC) globally and on the hard group.

Output:
  results/phase7_hardcase.json
  figures/RQ_hardcase_fpr.png         (FPR by group: easy vs hard)
  figures/RQ_hardcase_reliability.png (improvement concentrated in the uncertain band)

Run:  python phase7_hard_case_analysis.py --seeds 10
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# "hard" band = p_base uncertain within this range
HARD_LO, HARD_HI = 0.3, 0.7


class ContextGatingModule(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid())
        self.ctx = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, p_base):
        alpha = self.gate(x).squeeze(1)
        p_ctx = self.ctx(x).squeeze(1)
        return alpha * p_base + (1 - alpha) * p_ctx, alpha, p_ctx


# ── metric helpers ───────────────────────────────────────────────────────────
def tpr_at(y, p, thr=0.5):
    pr = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) else 0.0


def fpr_at_05(y, p):
    pr = (np.asarray(p) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def fpr_at_recall(y, p, target_tpr):
    fpr, tpr, _ = roc_curve(y, p)
    ok = tpr >= target_tpr - 1e-9
    return float(fpr[np.argmax(ok)]) if ok.any() else None


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(set(y.tolist())) > 1 else float("nan")


# ── train CGM on the original split, return p_final + alpha for ALL N ─────────
def train_and_infer(p_base, ctx, y, splits, seed):
    tr, va, te = splits == 0, splits == 1, splits == 2
    X = np.concatenate([p_base.reshape(-1, 1), ctx], axis=1)
    scaler = StandardScaler().fit(X[tr])
    Xn = scaler.transform(X).astype(np.float32)

    torch.manual_seed(seed); np.random.seed(seed)
    model = ContextGatingModule(X.shape[1])
    crit = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                     factor=0.5, patience=10)
    t = lambda a: torch.tensor(a)
    Xtr, ytr, pbtr = t(Xn[tr]), t(y[tr].astype(np.float32)), t(p_base[tr].astype(np.float32))
    Xva, yva, pbva = t(Xn[va]), t(y[va].astype(np.float32)), t(p_base[va].astype(np.float32))

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
        pf, alpha, p_ctx = model(t(Xn), t(p_base.astype(np.float32)))
    return pf.numpy(), alpha.numpy(), p_ctx.numpy()


def group_metrics(y, pb, pf, mask):
    """Compute FPR@0.5, FPR@matched-recall, AUC for baseline vs CGM on subset mask."""
    ys, pbs, pfs = y[mask], pb[mask], pf[mask]
    if len(ys) == 0 or len(set(ys.tolist())) < 2:
        return None
    t0 = tpr_at(ys, pbs)  # baseline recall as the anchor
    return dict(
        n=int(mask.sum()),
        fpr05_b=fpr_at_05(ys, pbs), fpr05_a=fpr_at_05(ys, pfs),
        mr_b=fpr_at_recall(ys, pbs, t0), mr_a=fpr_at_recall(ys, pfs, t0),
        auc_b=safe_auc(ys, pbs), auc_a=safe_auc(ys, pfs),
    )


def main(args):
    p_base = np.load(CACHE / "p_base.npy")
    ctx = np.concatenate([np.load(CACHE / "z_crowd.npy"),
                          np.load(CACHE / "z_light.npy"),
                          np.load(CACHE / "z_motion.npy")], axis=1)
    y = np.load(CACHE / "labels.npy").astype(int)
    splits = np.load(CACHE / "splits.npy")
    te = splits == 2

    print("=" * 70)
    print(f"PHASE 7 — Hard-case analysis (RWF, original split, {args.seeds} seeds)")
    print(f"  test={te.sum()} clips | hard band: {HARD_LO}<p_base<{HARD_HI}")
    print("=" * 70)

    # train many seeds, average p_final for plotting + collect metrics per seed
    all_overall, all_easy, all_hard = [], [], []
    pf_accum = np.zeros_like(p_base)
    alpha_accum = np.zeros_like(p_base)
    for s in range(args.seeds):
        pf, alpha, _ = train_and_infer(p_base, ctx, y, splits, s)
        pf_accum += pf; alpha_accum += alpha
        hard = te & (p_base > HARD_LO) & (p_base < HARD_HI)
        easy = te & ~((p_base > HARD_LO) & (p_base < HARD_HI))
        all_overall.append(group_metrics(y, p_base, pf, te))
        all_easy.append(group_metrics(y, p_base, pf, easy))
        all_hard.append(group_metrics(y, p_base, pf, hard))
    pf_mean = pf_accum / args.seeds
    alpha_mean = alpha_accum / args.seeds

    def agg(lst, k):
        vals = [d[k] for d in lst if d is not None and d[k] is not None
                and not (isinstance(d[k], float) and np.isnan(d[k]))]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)

    def report(name, lst):
        if all(d is None for d in lst):
            print(f"\n  [{name}] (not enough samples)"); return None
        n = agg(lst, "n")[0]
        print(f"\n  [{name}]  n≈{n:.0f}")
        for label, kb, ka in [("FPR@0.5", "fpr05_b", "fpr05_a"),
                              ("FPR@matched-recall", "mr_b", "mr_a"),
                              ("AUC-ROC", "auc_b", "auc_a")]:
            mb, sb = agg(lst, kb); ma, sa = agg(lst, ka)
            print(f"    {label:20} {mb:6.4f}±{sb:.3f} -> {ma:6.4f}±{sa:.3f}  Δ={ma-mb:+.4f}")
        return dict(n=n,
                    fpr05=(agg(lst,'fpr05_b')[0], agg(lst,'fpr05_a')[0]),
                    mr=(agg(lst,'mr_b')[0], agg(lst,'mr_a')[0]),
                    auc=(agg(lst,'auc_b')[0], agg(lst,'auc_a')[0]))

    r_overall = report("WHOLE TEST", all_overall)
    r_easy = report("EASY cases (p_base clear)", all_easy)
    r_hard = report("HARD cases (p_base uncertain)", all_hard)

    # context-only AUC (global & on hard cases) — evidence context has signal
    tr = splits == 0
    sc = StandardScaler().fit(ctx[tr]); Cn = sc.transform(ctx)
    lr = LogisticRegression(max_iter=1000).fit(Cn[tr], y[tr])
    pctx_lr = lr.predict_proba(Cn)[:, 1]
    auc_ctx_all = safe_auc(y[te], pctx_lr[te])
    hard = te & (p_base > HARD_LO) & (p_base < HARD_HI)
    auc_ctx_hard = safe_auc(y[hard], pctx_lr[hard])
    auc_pb_hard = safe_auc(y[hard], p_base[hard])
    print(f"\n  [Context-only LogReg] AUC whole test={auc_ctx_all:.4f} | "
          f"on hard cases={auc_ctx_hard:.4f} (p_base hard cases={auc_pb_hard:.4f})")

    # ── save JSON ──
    out = dict(
        seeds=args.seeds, hard_band=[HARD_LO, HARD_HI],
        overall=r_overall, easy=r_easy, hard=r_hard,
        context_only_auc_test=float(auc_ctx_all),
        context_only_auc_hard=float(auc_ctx_hard),
        pbase_auc_hard=float(auc_pb_hard),
    )
    def to_native(o):
        if isinstance(o, dict): return {k: to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [to_native(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return o
    with open(RESULTS / "phase7_hardcase.json", "w") as f:
        json.dump(to_native(out), f, indent=2)
    print(f"\n  saved -> results/phase7_hardcase.json")

    # ── FIGURE 1: FPR@matched-recall by group ──
    if r_easy and r_hard:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        groups = ["Easy\n(confident)", "Hard\n(uncertain)", "Overall"]
        b = [r_easy["mr"][0], r_hard["mr"][0], r_overall["mr"][0]]
        a = [r_easy["mr"][1], r_hard["mr"][1], r_overall["mr"][1]]
        x = np.arange(len(groups)); w = 0.36
        ax.bar(x - w/2, b, w, label="X3D-S (baseline)", color="#9aa7b8")
        ax.bar(x + w/2, a, w, label="X3D-S + CGM", color="#2e6fb7")
        ax.set_xticks(x); ax.set_xticklabels(groups)
        ax.set_ylabel("FPR @ matched recall")
        ax.set_title("CGM reduces false alarms mainly on hard cases")
        ax.legend(frameon=False)
        for i, (bb, aa) in enumerate(zip(b, a)):
            ax.text(i - w/2, bb + 0.005, f"{bb:.3f}", ha="center", fontsize=8)
            ax.text(i + w/2, aa + 0.005, f"{aa:.3f}", ha="center", fontsize=8)
        fig.tight_layout(); fig.savefig(FIGURES / "RQ_hardcase_fpr.png", dpi=160)
        print(f"  saved -> figures/RQ_hardcase_fpr.png")

    # ── FIGURE 2: |p_final - p_base| vs p_base uncertainty ──
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(p_base[te], pf_mean[te] - p_base[te],
               c=y[te], cmap="coolwarm", s=14, alpha=0.6)
    ax.axvspan(HARD_LO, HARD_HI, color="orange", alpha=0.12, label="hard band")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("p_base (X3D-S confidence)")
    ax.set_ylabel("p_final − p_base  (CGM correction)")
    ax.set_title("CGM intervenes where the detector is uncertain")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "RQ_hardcase_reliability.png", dpi=160)
    print(f"  saved -> figures/RQ_hardcase_reliability.png")

    # ── verdict ──
    print("\n" + "=" * 70)
    if r_hard and r_easy:
        d_hard = r_hard["mr"][0] - r_hard["mr"][1]
        d_easy = r_easy["mr"][0] - r_easy["mr"][1]
        print(f"  FPR@matched-recall reduction:  hard cases={d_hard:+.4f}  |  easy cases={d_easy:+.4f}")
        if d_hard > d_easy and d_hard > 0:
            print("  => CONFIRMED: CGM's benefit is CONCENTRATED in the hard group. "
                  "The Direction 1 story holds.")
        else:
            print("  => Benefit NOT clearly concentrated on hard cases — revisit the band or features.")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    main(args)
