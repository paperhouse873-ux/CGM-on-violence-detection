"""
Phase 10 — PAPER figure set (synced with the new results)
========================================================
Replaces the misleading phase4b figures (FPR 26.1% @0.5, zero-shot "success")
with an honest, style-consistent set following 3 pillars + 1 limitation:

  FIG1  honest_rq1_xferable   — X3D-S: @0.5 vs matched-recall + AUC + hard-case
  FIG2  multidetector_hardcase— hard-case FPR/AUC on X3D-S/SlowFast/MViT
  FIG3  headroom_curve        — (hero) CGM benefit vs detector weakness
  FIG4  rlvs_transfer         — zero-shot collapse vs in-domain works (+ alpha)

Reads: cache/p_base*.npy, cache/z_*.npy, cache/labels,splits ; cache/rlvs/* ;
     checkpoints/cgm_e4.pth + cgm_scaler.pkl ; results/*.json
All CGMs trained with the STRICT protocol (select by val AUC, many seeds,
matched-recall + AUC + Δα) — matching phase7/phase9.

Run:  python phase10_paper_figures.py --seeds 10
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE = Path("cache")
RLVS = CACHE / "rlvs"
CKPT = Path("checkpoints")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
FIG = Path("figures"); FIG.mkdir(exist_ok=True)

HARD_LO, HARD_HI = 0.3, 0.7

# ── consistent styling ───────────────────────────────────────────────────────
C_BASE = "#9aa7b8"      # baseline (blue-gray)
C_CGM  = "#2e6fb7"      # +CGM (dark blue)
C_POS  = "#2e7d32"      # good
C_NEG  = "#c62828"      # bad
C_VIO  = "#d6604d"; C_NOR = "#4393c3"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
})


class CGM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(32, 1), nn.Sigmoid())
        self.ctx = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x, pb):
        a = self.gate(x).squeeze(1); pc = self.ctx(x).squeeze(1)
        return a * pb + (1 - a) * pc, a, pc


def tpr_at(y, p, thr=0.5):
    pr = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return tp / (tp + fn) if (tp + fn) else 0.0


def fpr_at_05(y, p):
    pr = (np.asarray(p) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pr, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def fpr_at_recall(y, p, t):
    fpr, tpr, _ = roc_curve(y, p); ok = tpr >= t - 1e-9
    return float(fpr[np.argmax(ok)]) if ok.any() else np.nan


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(set(list(y))) > 1 else np.nan


def train_cgm(Xtr, ytr, pbtr, Xva, yva, pbva, d, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = CGM(d); crit = nn.BCELoss()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    t = lambda a: torch.tensor(a)
    Xtr, ytr, pbtr = t(Xtr), t(ytr), t(pbtr); Xva, pbva = t(Xva), t(pbva)
    best, bs, wait = -1, None, 0
    for _ in range(300):
        m.train(); pf, _, _ = m(Xtr, pbtr); loss = crit(pf, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pfv, _, _ = m(Xva, pbva)
        va = safe_auc(yva, pfv.numpy()); sch.step(0 if np.isnan(va) else va)
        if not np.isnan(va) and va > best:
            best, wait = va, 0; bs = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
        if wait >= 30:
            break
    m.load_state_dict(bs); m.eval()
    return m


def savefig(fig, name):
    p = FIG / name
    fig.savefig(p); plt.close(fig)
    print(f"    saved -> {p}")


# ════════════════════════════════════════════════════════════════════════════
# Compute p_final + alpha (RWF) on the original split, averaged over many seeds for figures
# ════════════════════════════════════════════════════════════════════════════

def rwf_infer(p_base, ctx, y, splits, seeds):
    tr, va = splits == 0, splits == 1
    X = np.concatenate([p_base.reshape(-1, 1), ctx], axis=1)
    sc = StandardScaler().fit(X[tr]); Xn = sc.transform(X).astype(np.float32)
    pf_acc = np.zeros(len(y), np.float64); al_acc = np.zeros(len(y), np.float64)
    for s in range(seeds):
        m = train_cgm(Xn[tr], y[tr].astype(np.float32), p_base[tr].astype(np.float32),
                      Xn[va], y[va], p_base[va].astype(np.float32), X.shape[1], s)
        with torch.no_grad():
            pf, a, _ = m(torch.tensor(Xn), torch.tensor(p_base.astype(np.float32)))
        pf_acc += pf.numpy(); al_acc += a.numpy()
    return pf_acc / seeds, al_acc / seeds


def detector_metrics(p_base, pf, alpha, y, splits):
    """Return dict of overall + hard metrics on the TEST split."""
    te = splits == 2
    hard = te & (p_base > HARD_LO) & (p_base < HARD_HI)
    out = {}
    for tag, mask in [("overall", te), ("hard", hard)]:
        ys, pbs, pfs = y[mask], p_base[mask], pf[mask]
        if mask.sum() == 0 or len(set(ys.tolist())) < 2:
            out[tag] = None; continue
        t0 = tpr_at(ys, pbs)
        out[tag] = dict(
            n=int(mask.sum()),
            fpr_b=fpr_at_recall(ys, pbs, t0), fpr_a=fpr_at_recall(ys, pfs, t0),
            auc_b=safe_auc(ys, pbs), auc_a=safe_auc(ys, pfs))
    yte = y[te]
    out["d_alpha"] = abs(alpha[te][yte == 1].mean() - alpha[te][yte == 0].mean())
    return out


# ════════════════════════════════════════════════════════════════════════════
# RLVS: zero-shot (CGM RWF) and in-domain (train on RLVS)
# ════════════════════════════════════════════════════════════════════════════

def rlvs_zero_shot():
    """Apply the CGM trained on RWF (cgm_e4.pth + scaler) to all of RLVS."""
    y = np.load(RLVS / "labels.npy").astype(int)
    pb = np.load(RLVS / "p_base.npy")
    Z = np.concatenate([np.load(RLVS / "z_crowd.npy"),
                        np.load(RLVS / "z_light.npy"),
                        np.load(RLVS / "z_motion.npy")], axis=1)
    ck = torch.load(CKPT / "cgm_e4.pth", map_location="cpu", weights_only=False)
    m = CGM(ck["input_dim"]); m.load_state_dict(ck["model_state_dict"]); m.eval()
    sc = pickle.load(open(CKPT / "cgm_scaler.pkl", "rb"))
    X = np.concatenate([pb.reshape(-1, 1), Z], axis=1)
    Xn = sc.transform(X).astype(np.float32)
    with torch.no_grad():
        pf, a, _ = m(torch.tensor(Xn), torch.tensor(pb.astype(np.float32)))
    pf, a = pf.numpy(), a.numpy()
    t0 = tpr_at(y, pb)
    return dict(
        fpr_b=fpr_at_recall(y, pb, t0), fpr_a=fpr_at_recall(y, pf, t0),
        auc_b=safe_auc(y, pb), auc_a=safe_auc(y, pf),
        d_alpha=abs(a[y == 1].mean() - a[y == 0].mean()),
        alpha=a, y=y)


def rlvs_in_domain(seeds):
    """Train CGM on RLVS (stratified 60/20/20), pool test across seeds for alpha."""
    y = np.load(RLVS / "labels.npy").astype(int)
    pb = np.load(RLVS / "p_base.npy")
    Z = np.concatenate([np.load(RLVS / "z_crowd.npy"),
                        np.load(RLVS / "z_light.npy"),
                        np.load(RLVS / "z_motion.npy")], axis=1)
    X = np.concatenate([pb.reshape(-1, 1), Z], axis=1)
    fb, fa, ab, aa, das = [], [], [], [], []
    al_all, y_all = [], []
    for s in range(seeds):
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=s)
        tr, va = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=s)
        sc = StandardScaler().fit(X[tr]); Xn = sc.transform(X).astype(np.float32)
        m = train_cgm(Xn[tr], y[tr].astype(np.float32), pb[tr].astype(np.float32),
                      Xn[va], y[va], pb[va].astype(np.float32), X.shape[1], s)
        with torch.no_grad():
            pf, a, _ = m(torch.tensor(Xn[te]), torch.tensor(pb[te].astype(np.float32)))
        pf, a = pf.numpy(), a.numpy(); yte = y[te]
        t0 = tpr_at(yte, pb[te])
        fb.append(fpr_at_recall(yte, pb[te], t0)); fa.append(fpr_at_recall(yte, pf, t0))
        ab.append(safe_auc(yte, pb[te])); aa.append(safe_auc(yte, pf))
        das.append(abs(a[yte == 1].mean() - a[yte == 0].mean()))
        al_all.append(a); y_all.append(yte)
    return dict(
        fpr_b=np.nanmean(fb), fpr_a=np.nanmean(fa),
        auc_b=np.nanmean(ab), auc_a=np.nanmean(aa),
        d_alpha=np.mean(das),
        alpha=np.concatenate(al_all), y=np.concatenate(y_all))


# ════════════════════════════════════════════════════════════════════════════
# FIG1 — Honest RQ1 (X3D-S): illusory @0.5 vs matched-recall + AUC + hard-case
# ════════════════════════════════════════════════════════════════════════════

def fig1_honest_rq1(det):
    """det = X3D-S metrics dict (from detector_metrics) + @0.5 values."""
    ov, hd = det["overall"], det["hard"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.suptitle("Context gating: apparent gain at 0.5 is recalibration; "
                 "the real gain is on hard cases (X3D-S, RWF test)",
                 fontsize=12.5, fontweight="bold")

    # (a) FPR: @0.5 vs @matched-recall (whole test)
    ax = axes[0]
    groups = ["FPR @0.5\n(naive)", "FPR @matched\nrecall (fair)"]
    b = [det["fpr05_b"], ov["fpr_b"]]; a = [det["fpr05_a"], ov["fpr_a"]]
    x = np.arange(2); w = 0.36
    r1 = ax.bar(x - w/2, b, w, label="X3D-S", color=C_BASE, edgecolor="black", lw=0.6)
    r2 = ax.bar(x + w/2, a, w, label="X3D-S + CGM", color=C_CGM, edgecolor="black", lw=0.6)
    ax.bar_label(r1, fmt="%.3f", fontsize=8); ax.bar_label(r2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("False Positive Rate"); ax.set_title("(a) Overall test set")
    ax.legend()
    ax.text(0.5, 0.95, "naive ↓ but fair ≈ → recalibration",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            style="italic", color=C_NEG)

    # (b) hard-case: FPR@matched-recall + AUC (the real evidence)
    ax = axes[1]
    labels = ["FPR@mr\n(hard)", "AUC\n(hard)"]
    b = [hd["fpr_b"], hd["auc_b"]]; a = [hd["fpr_a"], hd["auc_a"]]
    x = np.arange(2)
    r1 = ax.bar(x - w/2, b, w, label="X3D-S", color=C_BASE, edgecolor="black", lw=0.6)
    r2 = ax.bar(x + w/2, a, w, label="X3D-S + CGM", color=C_POS, edgecolor="black", lw=0.6)
    ax.bar_label(r1, fmt="%.3f", fontsize=8); ax.bar_label(r2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(f"(b) Hard cases (uncertain, n={hd['n']})")
    ax.set_ylabel("Score"); ax.legend()
    ax.text(0.5, 0.95, "FPR ↓ AND AUC ↑ → genuine",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            style="italic", color=C_POS)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "FIG1_honest_rq1.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG2 — Multi-detector hard-case (model-agnostic)
# ════════════════════════════════════════════════════════════════════════════

def fig2_multidetector(dets):
    """dets = list of (name, metrics)."""
    names = [d[0] for d in dets]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("Model-agnostic: CGM improves hard cases across architectures "
                 "(RWF test, matched recall)", fontsize=12.5, fontweight="bold")

    x = np.arange(len(names)); w = 0.36
    # (a) hard-case FPR@matched-recall
    ax = axes[0]
    fb = [d[1]["hard"]["fpr_b"] for d in dets]
    fa = [d[1]["hard"]["fpr_a"] for d in dets]
    r1 = ax.bar(x - w/2, fb, w, label="base", color=C_BASE, edgecolor="black", lw=0.6)
    r2 = ax.bar(x + w/2, fa, w, label="+CGM", color=C_CGM, edgecolor="black", lw=0.6)
    ax.bar_label(r1, fmt="%.2f", fontsize=8); ax.bar_label(r2, fmt="%.2f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("FPR @ matched recall (hard cases)")
    ax.set_title("(a) Hard-case false alarms"); ax.legend()

    # (b) hard-case AUC
    ax = axes[1]
    ab = [d[1]["hard"]["auc_b"] for d in dets]
    aa = [d[1]["hard"]["auc_a"] for d in dets]
    r1 = ax.bar(x - w/2, ab, w, label="base", color=C_BASE, edgecolor="black", lw=0.6)
    r2 = ax.bar(x + w/2, aa, w, label="+CGM", color=C_POS, edgecolor="black", lw=0.6)
    ax.bar_label(r1, fmt="%.2f", fontsize=8); ax.bar_label(r2, fmt="%.2f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("AUC (hard cases)")
    ax.set_title("(b) Hard-case ranking quality"); ax.legend()
    # note: number of hard cases per detector
    note = "  ".join(f"{d[0]}: n_hard={d[1]['hard']['n']}" for d in dets)
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    savefig(fig, "FIG2_multidetector_hardcase.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG3 — Headroom curve (HERO)
# ════════════════════════════════════════════════════════════════════════════

def fig3_headroom(points):
    """points = list of (name, dataset, weakness, benefit)."""
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    xs = [p[2] for p in points]; ys = [p[3] for p in points]
    # color: RWF blue, RLVS orange-red
    cols = [C_VIO if "RLVS" in p[1] else C_CGM for p in points]
    ax.scatter(xs, ys, s=110, c=cols, edgecolor="black", zorder=3)
    for nm, ds, x, yv in points:
        if "RLVS" in ds:
            # highest point (red dot): label BELOW the dot so it doesn't cover the title
            ax.annotate(f"{nm}\n({ds})", (x, yv), textcoords="offset points",
                        xytext=(4, -10), ha="right", va="top", fontsize=8.5)
        else:
            ax.annotate(f"{nm}\n({ds})", (x, yv), textcoords="offset points",
                        xytext=(8, 6), fontsize=8.5)
    if len(xs) >= 2:
        z = np.polyfit(xs, ys, 1)
        xx = np.linspace(min(xs) - 0.005, max(xs) + 0.01, 50)
        ax.plot(xx, np.polyval(z, xx), "--", color="gray",
                label=f"trend (slope={z[0]:.2f})")
        # correlation coefficient
        r = np.corrcoef(xs, ys)[0, 1]
        ax.text(0.03, 0.95, f"Pearson r = {r:.2f}", transform=ax.transAxes,
                fontsize=9, va="top")
        ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.90))
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Baseline weakness  (1 − AUC)")
    ax.set_ylabel("CGM benefit  (FPR reduction @ matched recall)")
    ax.set_title("Headroom relationship:\nweaker / domain-shifted detector → larger CGM benefit")
    # leave extra top margin so the highest dot isn't right against the title
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.02)
    fig.tight_layout()
    savefig(fig, "FIG3_headroom_curve.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG4 — RLVS transfer: zero-shot collapse vs in-domain works
# ════════════════════════════════════════════════════════════════════════════

def fig4_rlvs_transfer(zs, idm):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Cross-domain transfer to RLVS: zero-shot fails (gate collapse), "
                 "in-domain CGM works", fontsize=12.5, fontweight="bold")

    # (a) FPR@matched-recall
    ax = axes[0]
    x = np.arange(2); w = 0.36
    b = [zs["fpr_b"], idm["fpr_b"]]; a = [zs["fpr_a"], idm["fpr_a"]]
    r1 = ax.bar(x - w/2, b, w, label="X3D-S (base)", color=C_BASE, edgecolor="black", lw=0.6)
    r2 = ax.bar(x + w/2, a, w, label="+CGM", color=C_CGM, edgecolor="black", lw=0.6)
    ax.bar_label(r1, fmt="%.2f", fontsize=8); ax.bar_label(r2, fmt="%.2f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["Zero-shot\n(CGM/RWF)", "In-domain\n(CGM/RLVS)"])
    ax.set_ylabel("FPR @ matched recall"); ax.set_title("(a) False alarms")
    ax.legend(fontsize=8)

    # (b) AUC
    ax = axes[1]
    b = [zs["auc_b"], idm["auc_b"]]; a = [zs["auc_a"], idm["auc_a"]]
    r1 = ax.bar(x - w/2, b, w, label="base", color=C_BASE, edgecolor="black", lw=0.6)
    r2 = ax.bar(x + w/2, a, w, label="+CGM", color=C_POS, edgecolor="black", lw=0.6)
    ax.bar_label(r1, fmt="%.3f", fontsize=8); ax.bar_label(r2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["Zero-shot", "In-domain"])
    ax.set_ylabel("AUC-ROC"); ax.set_title("(b) Ranking quality"); ax.set_ylim(0.8, 1.0)
    ax.legend(fontsize=8)

    # (c) alpha distribution by class: collapse vs alive
    ax = axes[2]
    data = [
        zs["alpha"][zs["y"] == 0], zs["alpha"][zs["y"] == 1],
        idm["alpha"][idm["y"] == 0], idm["alpha"][idm["y"] == 1],
    ]
    pos = [1, 1.6, 3, 3.6]
    parts = ax.violinplot(data, positions=pos, widths=0.5, showmeans=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(C_NOR if i % 2 == 0 else C_VIO); pc.set_alpha(0.6)
    ax.set_xticks([1.3, 3.3])
    ax.set_xticklabels([f"Zero-shot\n|Δα|={zs['d_alpha']:.3f}",
                        f"In-domain\n|Δα|={idm['d_alpha']:.3f}"])
    ax.set_ylabel("gate α")
    ax.set_title("(c) Gate behaviour by class")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_NOR, alpha=0.6),
               plt.Rectangle((0, 0), 1, 1, color=C_VIO, alpha=0.6)]
    ax.legend(handles, ["Non-violent", "Violent"], fontsize=8, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    savefig(fig, "FIG4_rlvs_transfer.png")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    y = np.load(CACHE / "labels.npy").astype(int)
    splits = np.load(CACHE / "splits.npy")
    ctx = np.concatenate([np.load(CACHE / "z_crowd.npy"),
                          np.load(CACHE / "z_light.npy"),
                          np.load(CACHE / "z_motion.npy")], axis=1)

    DETS = [("X3D-S", CACHE / "p_base.npy"),
            ("SlowFast", CACHE / "p_base_slowfast_r50.npy"),
            ("MViT", CACHE / "p_base_mvit_base_16x4.npy")]

    print("=" * 64)
    print(f"PHASE 10 — paper figures ({args.seeds} seeds)")
    print("=" * 64)

    det_metrics = []
    for name, path in DETS:
        if not path.exists():
            print(f"  [skip] {name}: missing {path.name}"); continue
        pb = np.load(path)
        pf, al = rwf_infer(pb, ctx, y, splits, args.seeds)
        m = detector_metrics(pb, pf, al, y, splits)
        # add @0.5 for X3D-S (FIG1)
        te = splits == 2
        m["fpr05_b"] = fpr_at_05(y[te], pb[te])
        m["fpr05_a"] = fpr_at_05(y[te], pf[te])
        det_metrics.append((name, m))
        print(f"  {name}: hard FPR@mr {m['hard']['fpr_b']:.3f}->{m['hard']['fpr_a']:.3f}"
              f" | hard AUC {m['hard']['auc_b']:.3f}->{m['hard']['auc_a']:.3f}"
              f" | |Δα|={m['d_alpha']:.3f}")

    # RLVS
    print("  RLVS zero-shot ...")
    zs = rlvs_zero_shot()
    print(f"    FPR@mr {zs['fpr_b']:.3f}->{zs['fpr_a']:.3f} | AUC {zs['auc_b']:.3f}->{zs['auc_a']:.3f} | |Δα|={zs['d_alpha']:.3f}")
    print("  RLVS in-domain ...")
    idm = rlvs_in_domain(args.seeds)
    print(f"    FPR@mr {idm['fpr_b']:.3f}->{idm['fpr_a']:.3f} | AUC {idm['auc_b']:.3f}->{idm['auc_a']:.3f} | |Δα|={idm['d_alpha']:.3f}")

    # ── draw ──
    x3d = dict(det_metrics)["X3D-S"]
    fig1_honest_rq1(x3d)
    fig2_multidetector(det_metrics)

    # headroom points: RWF detectors + RLVS in-domain
    pts = []
    for name, m in det_metrics:
        wk = 1 - m["overall"]["auc_b"]
        bn = m["overall"]["fpr_b"] - m["overall"]["fpr_a"]
        pts.append((name, "RWF", wk, bn))
    pts.append(("X3D-S", "RLVS in-dom", 1 - idm["auc_b"], idm["fpr_b"] - idm["fpr_a"]))
    fig3_headroom(pts)

    fig4_rlvs_transfer(zs, idm)

    # save the summary table of numbers for the paper
    def native(o):
        if isinstance(o, dict): return {k: native(v) for k, v in o.items() if k not in ("alpha", "y")}
        if isinstance(o, (list, tuple)): return [native(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, float) and np.isnan(o): return None
        return o
    summary = {"detectors": {n: native(m) for n, m in det_metrics},
               "rlvs_zero_shot": native(zs), "rlvs_in_domain": native(idm),
               "headroom_points": [list(p) for p in pts]}
    json.dump(native(summary), open(RESULTS / "phase10_summary.json", "w"), indent=2)
    print("  saved -> results/phase10_summary.json")
    print("=" * 64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    main(args)
