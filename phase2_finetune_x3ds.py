import json
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix,
)

import sys
sys.path.append(str(Path(__file__).parent))
from phase1_dataset import RWF2000Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load X3D-S + fix head
# ═══════════════════════════════════════════════════════════════════════════

def build_x3ds_model() -> nn.Module:
    print("\n  Loading X3D-S pretrained Kinetics-400...")
    from pytorchvideo.models.hub import x3d_s
    model = x3d_s(pretrained=True)

    in_features = model.blocks[-1].proj.in_features
    print(f"  Original head: Linear({in_features} -> 400) + Softmax")

    # FIX 1: change proj to 1 output
    model.blocks[-1].proj = nn.Linear(in_features, 1)
    # FIX 2: REMOVE the Softmax inside the head (the cause of FPR=1.0)
    model.blocks[-1].activation = nn.Identity()
    # some versions name it differently — set both to be safe
    if hasattr(model.blocks[-1], "act"):
        model.blocks[-1].act = nn.Identity()

    print(f"  New head: Linear({in_features} -> 1), Softmax REMOVED")

    model = model.to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 2. Metrics
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


# ═══════════════════════════════════════════════════════════════════════════
# 3. Train one epoch
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    pbar = tqdm(loader, desc=f"  [Train E{epoch:02d}]", unit="batch")
    for videos, labels in pbar:
        videos = videos.to(DEVICE)
        labels = labels.float().to(DEVICE)

        optimizer.zero_grad()
        logits = model(videos).squeeze(1)       # NO sigmoid here
        loss = criterion(logits, labels)        # BCEWithLogitsLoss handles it
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits.detach())  # just for the metric
        total_loss += loss.item()
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    m = compute_metrics(all_labels, all_probs)
    m["loss"] = round(total_loss / len(loader), 4)
    return m


# ═══════════════════════════════════════════════════════════════════════════
# 4. Evaluate
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, criterion, desc="Val"):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for videos, labels in tqdm(loader, desc=f"  [{desc}]", unit="batch"):
        videos = videos.to(DEVICE)
        labels_f = labels.float().to(DEVICE)

        logits = model(videos).squeeze(1)
        loss = criterion(logits, labels_f)
        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    m = compute_metrics(all_labels, all_probs)
    m["loss"] = round(total_loss / len(loader), 4)
    return m, all_labels, all_probs


# ═══════════════════════════════════════════════════════════════════════════
# 5. Main train loop
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)

    print("\n  Loading datasets...")
    train_ds = RWF2000Dataset(args.root, args.split, "train", augment=True)
    val_ds   = RWF2000Dataset(args.root, args.split, "val",   augment=False)
    test_ds  = RWF2000Dataset(args.root, args.split, "test",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    model = build_x3ds_model()

    # FIX 3: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    print(f"\n  {'='*55}")
    print(f"  Fine-tuning X3D-S — {args.epochs} epochs")
    print(f"  Batch: {args.batch_size} | LR: {args.lr} | BCEWithLogitsLoss")
    print(f"  {'='*55}\n")

    best_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, epoch)
        val_m, _, _ = evaluate(model, val_loader, criterion, "Val")
        elapsed = time.time() - t0
        scheduler.step(val_m["f1"])

        print(f"\n  Epoch {epoch:02d}/{args.epochs} ({elapsed:.0f}s)")
        print(f"  Train — loss:{train_m['loss']:.4f}  acc:{train_m['accuracy']:.4f}  "
              f"f1:{train_m['f1']:.4f}  fpr:{train_m['fpr']:.4f}")
        print(f"  Val   — loss:{val_m['loss']:.4f}  acc:{val_m['accuracy']:.4f}  "
              f"f1:{val_m['f1']:.4f}  fpr:{val_m['fpr']:.4f}")

        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_f1, "val_metrics": val_m,
            }, ckpt_dir / "x3ds_best.pth")
            print(f"  ** Saved best checkpoint (F1={best_f1:.4f}) **")

    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict()},
               ckpt_dir / "x3ds_last.pth")
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  {'='*55}")
    print(f"  Evaluating best model on TEST SET -> E0 baseline")
    print(f"  {'='*55}")

    ckpt = torch.load(ckpt_dir / "x3ds_best.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    test_m, _, _ = evaluate(model, test_loader, criterion, "Test")

    print(f"\n  === E0 Baseline Results ===")
    print(f"  Accuracy : {test_m['accuracy']:.4f}")
    print(f"  F1 Score : {test_m['f1']:.4f}")
    print(f"  AUC-ROC  : {test_m['auc_roc']:.4f}")
    print(f"  FPR      : {test_m['fpr']:.4f}  <- primary metric to beat")
    print(f"  FNR      : {test_m['fnr']:.4f}  <- must not increase later")
    print(f"  TP:{test_m['tp']} TN:{test_m['tn']} FP:{test_m['fp']} FN:{test_m['fn']}")

    with open(results_dir / "E0_baseline.json", "w") as f:
        json.dump({"experiment": "E0", "metrics": test_m,
                   "n_test": len(test_ds), "best_epoch": ckpt["epoch"]},
                  f, indent=2)
    print(f"\n  Saved -> results/E0_baseline.json")
    print(f"  Phase 2 DONE.\n")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",         type=str, required=True)
    parser.add_argument("--split",        type=str, required=True)
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)