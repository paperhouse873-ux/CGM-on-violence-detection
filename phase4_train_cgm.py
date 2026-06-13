import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Context Gating Module — 962 params
# ═══════════════════════════════════════════════════════════════════════════

class ContextGatingModule(nn.Module):
    """
    The heart of the pipeline.
    Input:  13-dim context vector (or subset)
    Output: p_final = α · p_base + (1-α) · p_ctx
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # MLP-gate → α: how much to trust X3D-S
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        # MLP-ctx → p_ctx: corrected probability from context
        self.ctx = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, p_base):
        alpha = self.gate(x).squeeze(1)   # (B,)
        p_ctx = self.ctx(x).squeeze(1)    # (B,)
        p_final = alpha * p_base + (1 - alpha) * p_ctx
        return p_final, alpha, p_ctx

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# 2. Load cached data from Phase 3
# ═══════════════════════════════════════════════════════════════════════════

def load_cache():
    cache = Path("cache")
    data = {
        "p_base":  np.load(cache / "p_base.npy"),       # (N,)
        "z_crowd": np.load(cache / "z_crowd.npy"),       # (N,4)
        "z_light": np.load(cache / "z_light.npy"),       # (N,4)
        "z_motion":np.load(cache / "z_motion.npy"),      # (N,4)
        "labels":  np.load(cache / "labels.npy"),         # (N,)
        "splits":  np.load(cache / "splits.npy"),         # (N,)
        "context": np.load(cache / "context_13dim.npy"),  # (N,13)
    }
    print(f"  Loaded cache: {data['p_base'].shape[0]} samples")
    return data


def split_data(data, split_id):
    """Return the subset for a split: 0=train, 1=val, 2=test"""
    mask = data["splits"] == split_id
    return {k: v[mask] for k, v in data.items()}


def build_context_vector(data, streams):
    """
    Build the context vector from the selected streams.
    streams: list of 'crowd', 'light', 'motion'
    p_base (1-dim) is always the first feature.
    """
    parts = [data["p_base"].reshape(-1, 1)]  # p_base always included
    if "crowd"  in streams: parts.append(data["z_crowd"])
    if "light"  in streams: parts.append(data["z_light"])
    if "motion" in streams: parts.append(data["z_motion"])
    X = np.concatenate(parts, axis=1)

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return X, scaler


# ═══════════════════════════════════════════════════════════════════════════
# 3. Train loop for CGM
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(labels, probs, threshold=0.5):
    preds = [1 if p >= threshold else 0 for p in probs]
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "f1":       round(f1_score(labels, preds), 4),
        "auc_roc":  round(roc_auc_score(labels, probs), 4),
        "fpr":      round(fpr, 4),
        "fnr":      round(fnr, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def train_cgm(
    train_data, val_data, test_data,
    streams: list,
    pos_weight: float = 1.0,
    epochs: int = 100,
    lr: float = 1e-3,
    exp_name: str = "E?",
):
    """Train CGM on one specific ablation configuration."""
    # Build context vectors
    X_raw, scaler = build_context_vector(train_data, streams)
    scaler.fit(X_raw)

    X_train = torch.tensor(scaler.transform(X_raw), dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(train_data["labels"], dtype=torch.float32).to(DEVICE)
    p_base_train = torch.tensor(train_data["p_base"], dtype=torch.float32).to(DEVICE)

    X_val_raw, _ = build_context_vector(val_data, streams)
    X_val = torch.tensor(scaler.transform(X_val_raw), dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(val_data["labels"], dtype=torch.float32).to(DEVICE)
    p_base_val = torch.tensor(val_data["p_base"], dtype=torch.float32).to(DEVICE)

    X_test_raw, _ = build_context_vector(test_data, streams)
    X_test = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(test_data["labels"], dtype=torch.float32).to(DEVICE)
    p_base_test = torch.tensor(test_data["p_base"], dtype=torch.float32).to(DEVICE)

    input_dim = X_train.shape[1]
    print(f"\n  [{exp_name}] streams={streams}, input_dim={input_dim}, "
          f"pos_weight={pos_weight}")

    # Model
    model = ContextGatingModule(input_dim).to(DEVICE)
    print(f"  [{exp_name}] CGM params: {model.count_params()}")

    # Loss & optimizer
    weight = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10)

    best_val_f1 = 0.0
    best_state  = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        p_final, alpha, p_ctx = model(X_train, p_base_train)
        loss_raw = criterion(p_final, y_train)
        # Apply cost-sensitive weighting
        weights = torch.where(y_train == 1, weight, torch.ones(1).to(DEVICE))
        loss = (loss_raw * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Val ──
        model.eval()
        with torch.no_grad():
            p_final_val, alpha_val, _ = model(X_val, p_base_val)
        val_m = compute_metrics(
            y_val.cpu().tolist(), p_final_val.cpu().tolist()
        )
        scheduler.step(val_m["f1"])

        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{exp_name}] E{epoch:03d} — loss:{loss.item():.4f}  "
                  f"val_f1:{val_m['f1']:.4f}  val_fpr:{val_m['fpr']:.4f}  "
                  f"α_mean:{alpha_val.mean().item():.3f}")

        # Early stopping
        if patience_counter >= 25:
            print(f"  [{exp_name}] Early stop at epoch {epoch}")
            break

    # ── Test with best model ──
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_final_test, alpha_test, p_ctx_test = model(X_test, p_base_test)

    test_m = compute_metrics(
        y_test.cpu().tolist(), p_final_test.cpu().tolist()
    )

    # Alpha distribution (for interpretability analysis)
    alpha_np = alpha_test.cpu().numpy()
    test_m["alpha_mean"]   = round(float(alpha_np.mean()), 4)
    test_m["alpha_std"]    = round(float(alpha_np.std()), 4)
    test_m["alpha_violent"] = round(float(
        alpha_np[y_test.cpu().numpy() == 1].mean()), 4)
    test_m["alpha_normal"]  = round(float(
        alpha_np[y_test.cpu().numpy() == 0].mean()), 4)

    print(f"\n  [{exp_name}] TEST RESULTS:")
    print(f"  Acc:{test_m['accuracy']:.4f}  F1:{test_m['f1']:.4f}  "
          f"AUC:{test_m['auc_roc']:.4f}  "
          f"FPR:{test_m['fpr']:.4f}  FNR:{test_m['fnr']:.4f}")
    print(f"  α mean: {test_m['alpha_mean']:.3f} "
          f"(violent={test_m['alpha_violent']:.3f}, "
          f"normal={test_m['alpha_normal']:.3f})")

    return {
        "experiment": exp_name,
        "streams":    streams,
        "pos_weight": pos_weight,
        "input_dim":  input_dim,
        "cgm_params": model.count_params(),
        "test_metrics": test_m,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Run 6 Ablation Experiments
# ═══════════════════════════════════════════════════════════════════════════

def run_ablation():
    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)

    print(f"\n  {'='*60}")
    print(f"  Phase 4 — Context Gating Module Ablation Study")
    print(f"  {'='*60}")

    data = load_cache()
    train_d = split_data(data, 0)
    val_d   = split_data(data, 1)
    test_d  = split_data(data, 2)

    # E0: X3D-S only (from Phase 2)
    e0_path = results_dir / "E0_baseline.json"
    if e0_path.exists():
        with open(e0_path) as f:
            e0_data = json.load(f)
        e0_m = e0_data.get("metrics", {})
        print(f"\n  [E0] X3D-S only (loaded from Phase 2)")
        print(f"  Acc:{e0_m.get('accuracy','?')}  F1:{e0_m.get('f1','?')}  "
              f"FPR:{e0_m.get('fpr','?')}  FNR:{e0_m.get('fnr','?')}")
    else:
        # compute E0 directly from p_base
        test_probs = test_d["p_base"].tolist()
        test_labels = test_d["labels"].tolist()
        e0_m = compute_metrics(test_labels, test_probs)
        print(f"\n  [E0] X3D-S only (computed from p_base)")
        print(f"  Acc:{e0_m['accuracy']}  F1:{e0_m['f1']}  "
              f"FPR:{e0_m['fpr']}  FNR:{e0_m['fnr']}")

    # 5 ablation experiments
    experiments = [
        ("E1", ["crowd"],                     1.0),
        ("E2", ["light"],                     1.0),
        ("E3", ["motion"],                    1.0),
        ("E4", ["crowd", "light", "motion"],  1.0),
        ("E5", ["crowd", "light", "motion"],  3.0),
    ]

    all_results = [{"experiment": "E0", "streams": [],
                    "test_metrics": e0_m}]

    for name, streams, pw in experiments:
        result = train_cgm(
            train_d, val_d, test_d,
            streams=streams,
            pos_weight=pw,
            epochs=200,
            lr=1e-3,
            exp_name=name,
        )
        all_results.append(result)

    # ── Print comparison table ────────────────────────────────────────
    print(f"\n\n  {'='*70}")
    print(f"  ABLATION RESULTS COMPARISON")
    print(f"  {'='*70}")
    print(f"  {'Exp':<6} {'Streams':<26} {'Acc':>6} {'F1':>6} "
          f"{'AUC':>6} {'FPR':>6} {'FNR':>6}")
    print(f"  {'─'*70}")

    for r in all_results:
        m = r.get("test_metrics", {})
        streams_str = ", ".join(r.get("streams", [])) or "none (X3D only)"
        exp = r["experiment"]
        marker = " ←" if exp == "E4" else ""
        print(f"  {exp:<6} {streams_str:<26} "
              f"{m.get('accuracy','?'):>6} {m.get('f1','?'):>6} "
              f"{m.get('auc_roc','?'):>6} {m.get('fpr','?'):>6} "
              f"{m.get('fnr','?'):>6}{marker}")

    print(f"  {'─'*70}")

    # FPR improvement
    e0_fpr = e0_m.get("fpr", 0)
    e4_m = all_results[4].get("test_metrics", {})
    e4_fpr = e4_m.get("fpr", 0)
    if isinstance(e0_fpr, (int, float)) and isinstance(e4_fpr, (int, float)):
        improvement = e0_fpr - e4_fpr
        print(f"\n  FPR improvement (E0 → E4): "
              f"{e0_fpr:.4f} → {e4_fpr:.4f} "
              f"(Δ = {improvement:.4f}, "
              f"{improvement/e0_fpr*100:.1f}% relative reduction)")

    # Save
    with open(results_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # CSV for easy viewing
    import csv
    with open(results_dir / "ablation_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Exp", "Streams", "Acc", "F1", "AUC", "FPR", "FNR"])
        for r in all_results:
            m = r.get("test_metrics", {})
            w.writerow([
                r["experiment"],
                "+".join(r.get("streams", [])) or "X3D_only",
                m.get("accuracy", ""), m.get("f1", ""),
                m.get("auc_roc", ""), m.get("fpr", ""), m.get("fnr", ""),
            ])

    print(f"\n  Saved → results/ablation_results.json")
    print(f"  Saved → results/ablation_table.csv")
    print(f"\n  Phase 4 DONE.\n")


if __name__ == "__main__":
    run_ablation()