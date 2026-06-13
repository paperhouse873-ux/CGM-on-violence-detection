"""
Phase 6b — In-domain CGM on RLVS (control vs zero-shot)
============================================================
Question (RQ3 extended): is the gate collapse at zero-shot (Δα≈0.0006) due to
the NATURE of the CGM, or just DOMAIN SHIFT? If we train CGM directly on RLVS
(X3D-S still frozen, only learn the gate + ctx head on RLVS context-features),
does the gate "come back to life"?

Designed TIGHT to survive review:
  * Split RLVS into stratified train/val/test. TEST is never used to train CGM
    or fit the scaler -> no leak.
  * Scaler fit on RLVS train ONLY (in-domain, unlike zero-shot which uses RWF scaler).
  * Early-stop on val.
  * Report ON TEST: threshold 0.5, FPR @ same recall (matched-recall),
    Youden, AUC, and Δα (whether the gate separates classes).
  * Repeat over many SEEDS -> mean ± std (avoid cherry-pick).

Three configs compared on the same test set:
  (a) baseline  : p_base (X3D-S frozen, trained on RWF)
  (b) zero-shot : CGM trained on RWF  (checkpoints/cgm_e4.pth + RWF scaler)
  (c) in-domain : CGM trained on RLVS-train (this script)

Run:
  python phase6b_train_rlvs.py                 # 5 seeds, default
  python phase6b_train_rlvs.py --seeds 10 --epochs 300
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RLVS_CACHE = Path("cache/rlvs")
CKPT = Path("checkpoints")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)


# ── CGM (same as Phase 4 / Phase 6) ──────────────────────────────────────────
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


# ── metrics helpers ──────────────────────────────────────────────────────────
def metrics_at_threshold(y, probs, thr=0.5):
    preds = (np.asarray(probs) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    return dict(fpr=fpr, fnr=1 - tpr, tpr=tpr,
                acc=(tp + tn) / len(y),
                f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)


def fpr_at_recall(y, probs, target_tpr):
    """smallest FPR where TPR >= target_tpr (scan the whole ROC)."""
    fpr, tpr, thr = roc_curve(y, probs)
    ok = tpr >= target_tpr - 1e-9
    if not ok.any():
        return None
    return float(fpr[np.argmax(ok)])


# ── train CGM on one RLVS split ──────────────────────────────────────────────
def train_one(Xtr, ytr, pbtr, Xva, yva, pbva, input_dim,
              epochs=300, lr=1e-3, patience=30, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ContextGatingModule(input_dim).to(DEVICE)
    crit = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=10)

    def t(a): return torch.tensor(a).to(DEVICE)
    Xtr, ytr, pbtr = t(Xtr), t(ytr), t(pbtr)
    Xva, yva, pbva = t(Xva), t(yva), t(pbva)

    best_auc, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        pf, _, _ = model(Xtr, pbtr)
        loss = crit(pf, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pfv, _, _ = model(Xva, pbva)
        try:
            vauc = roc_auc_score(yva.cpu().numpy(), pfv.cpu().numpy())
        except ValueError:
            vauc = 0.0
        sched.step(vauc)
        if vauc > best_auc:
            best_auc, wait = vauc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model


def run_seed(p_base, Z, y, seed, epochs):
    # 60/20/20 stratified: train / val / test
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.20, stratify=y,
                                      random_state=seed)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.25, stratify=y[tr_idx],
                                      random_state=seed)  # 0.25*0.8 = 0.2

    X = np.concatenate([p_base.reshape(-1, 1), Z], axis=1)
    scaler = StandardScaler().fit(X[tr_idx])          # fit on RLVS train ONLY
    Xn = scaler.transform(X).astype(np.float32)

    model = train_one(
        Xn[tr_idx], y[tr_idx].astype(np.float32), p_base[tr_idx].astype(np.float32),
        Xn[va_idx], y[va_idx].astype(np.float32), p_base[va_idx].astype(np.float32),
        input_dim=X.shape[1], epochs=epochs, seed=seed)

    # ---- evaluate ON TEST ----
    with torch.no_grad():
        pf, alpha, p_ctx = model(torch.tensor(Xn[te_idx]).to(DEVICE),
                                 torch.tensor(p_base[te_idx].astype(np.float32)).to(DEVICE))
    pf = pf.cpu().numpy(); alpha = alpha.cpu().numpy()
    yte = y[te_idx].astype(int)
    pbte = p_base[te_idx]

    b05 = metrics_at_threshold(yte, pbte, 0.5)
    a05 = metrics_at_threshold(yte, pf, 0.5)
    auc_b = roc_auc_score(yte, pbte)
    auc_a = roc_auc_score(yte, pf)
    # matched recall = baseline recall @0.5
    fpr_b_mr = fpr_at_recall(yte, pbte, b05["tpr"])
    fpr_a_mr = fpr_at_recall(yte, pf,   b05["tpr"])
    d_alpha = abs(alpha[yte == 1].mean() - alpha[yte == 0].mean())

    return dict(
        fpr_b=b05["fpr"], fpr_a=a05["fpr"],
        fnr_b=b05["fnr"], fnr_a=a05["fnr"],
        acc_b=b05["acc"], acc_a=a05["acc"],
        f1_b=b05["f1"],   f1_a=a05["f1"],
        auc_b=auc_b,      auc_a=auc_a,
        fpr_b_mr=fpr_b_mr, fpr_a_mr=fpr_a_mr,
        d_alpha=d_alpha,
        alpha_v=float(alpha[yte == 1].mean()),
        alpha_n=float(alpha[yte == 0].mean()),
    )


def agg(runs, key):
    vals = [r[key] for r in runs if r[key] is not None]
    return float(np.mean(vals)), float(np.std(vals))


def main(args):
    y = np.load(RLVS_CACHE / "labels.npy").astype(int)
    p_base = np.load(RLVS_CACHE / "p_base.npy")
    Z = np.concatenate([np.load(RLVS_CACHE / "z_crowd.npy"),
                        np.load(RLVS_CACHE / "z_light.npy"),
                        np.load(RLVS_CACHE / "z_motion.npy")], axis=1)

    print("=" * 70)
    print(f"PHASE 6b — In-domain CGM on RLVS  ({args.seeds} seeds, test split 20%)")
    print(f"Device={DEVICE}  | clips={len(y)}  | input_dim={1 + Z.shape[1]}")
    print("=" * 70)

    runs = [run_seed(p_base, Z, y, s, args.epochs) for s in range(args.seeds)]

    def line(name, kb, ka, pct=False):
        mb, sb = agg(runs, kb); ma, sa = agg(runs, ka)
        d = ma - mb
        arrow = ""
        print(f"  {name:18}{mb:6.4f}±{sb:.3f}   {ma:6.4f}±{sa:.3f}   Δ={d:+.4f}{arrow}")

    print("\n  On TEST set (mean ± std over seeds):")
    print(f"  {'metric':18}{'baseline':>14}   {'+CGM(RLVS)':>14}")
    line("FPR @0.5",      "fpr_b", "fpr_a")
    line("FNR @0.5",      "fnr_b", "fnr_a")
    line("Accuracy @0.5", "acc_b", "acc_a")
    line("F1 @0.5",       "f1_b",  "f1_a")
    line("AUC-ROC",       "auc_b", "auc_a")
    line("FPR @matched-recall", "fpr_b_mr", "fpr_a_mr")

    da_m, da_s = agg(runs, "d_alpha")
    av_m, _ = agg(runs, "alpha_v"); an_m, _ = agg(runs, "alpha_n")
    print(f"\n  |Δα| (gate separates classes) = {da_m:.4f} ± {da_s:.4f}")
    print(f"  α_violent={av_m:.4f}  α_normal={an_m:.4f}")

    # ---- automatic verdict based on 2 REAL criteria ----
    auc_mb, _ = agg(runs, "auc_b"); auc_ma, _ = agg(runs, "auc_a")
    mr_b, _ = agg(runs, "fpr_b_mr"); mr_a, _ = agg(runs, "fpr_a_mr")
    print("\n" + "=" * 70)
    print("VERDICT (based on matched-recall FPR + AUC + Δα, NOT FPR@0.5):")
    cond_auc = auc_ma >= auc_mb - 0.002
    cond_mr  = mr_a < mr_b - 1e-4
    cond_gate = da_m > 0.02
    print(f"  [{'OK' if cond_mr  else 'XX'}] FPR@matched-recall really drops: "
          f"{mr_b:.4f} -> {mr_a:.4f}")
    print(f"  [{'OK' if cond_auc else 'XX'}] AUC doesn't drop: {auc_mb:.4f} -> {auc_ma:.4f}")
    print(f"  [{'OK' if cond_gate else 'XX'}] Gate alive (|Δα|>0.02): {da_m:.4f}")
    if cond_mr and cond_auc and cond_gate:
        print("\n  => IN-DOMAIN CGM REALLY WORKS. The 'gate needs domain data' narrative\n"
              "     is proven. Good to put in the paper.")
    elif cond_gate and (cond_mr or cond_auc):
        print("\n  => POSITIVE signal but not complete. Look closely at each criterion.")
    else:
        print("\n  => Even in-domain CGM does NOT really improve. The problem is in the\n"
              "     context features (low information), not just domain shift.")
    print("=" * 70)

    out = dict(
        n_seeds=args.seeds, n_clips=int(len(y)), test_frac=0.20,
        baseline={k.replace("_b", ""): agg(runs, k)[0]
                  for k in ["fpr_b", "fnr_b", "acc_b", "f1_b", "auc_b", "fpr_b_mr"]},
        in_domain_cgm={k.replace("_a", ""): agg(runs, k)[0]
                       for k in ["fpr_a", "fnr_a", "acc_a", "f1_a", "auc_a", "fpr_a_mr"]},
        delta_alpha=da_m, alpha_violent=av_m, alpha_normal=an_m,
        per_seed=runs,
    )
    def to_native(o):
        if isinstance(o, dict):
            return {k: to_native(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_native(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    with open(RESULTS / "phase6b_indomain_results.json", "w") as f:
        json.dump(to_native(out), f, indent=2)
    print("  saved -> results/phase6b_indomain_results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    main(args)
