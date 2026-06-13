"""
Make the train/val/test split (70/15/15, stratified) for RWF-2000.
Run: python split.py --root /path/to/RWF-2000 --seed 42

RWF-2000 only ships train (1600) and val (400); we merge all 2000 clips and
re-split into 1400/300/300, then dump split.json so every experiment uses the
exact same data (otherwise the comparison isn't fair).
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

LABEL_MAP = {
    "fight":    1,  # violent
    "nonFight": 0,  # non-violent
}


def collect_clips(root: Path) -> dict[str, list[str]]:
    """collect clips by label, ignore the original split name (train/val)"""
    clips_by_label: dict[str, list[str]] = defaultdict(list)

    for split_dir in ["train", "val"]:
        for label in ["fight", "nonFight"]:
            folder = root / split_dir / label
            if not folder.exists():
                continue
            for f in folder.iterdir():
                if f.suffix.lower() in VIDEO_EXTS:
                    # store relative path so it's portable
                    clips_by_label[label].append(
                        str(f.relative_to(root))
                    )

    return dict(clips_by_label)


def stratified_split(
    clips_by_label: dict[str, list[str]],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> dict:
    """stratified split — each class keeps the same train/val/test ratio"""
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}

    for label, clips in clips_by_label.items():
        shuffled = clips[:]
        rng.shuffle(shuffled)

        n  = len(shuffled)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        split["train"] += [{"path": p, "label": LABEL_MAP[label]}
                           for p in shuffled[:n_train]]
        split["val"]   += [{"path": p, "label": LABEL_MAP[label]}
                           for p in shuffled[n_train:n_train + n_val]]
        split["test"]  += [{"path": p, "label": LABEL_MAP[label]}
                           for p in shuffled[n_train + n_val:]]

    # reshuffle so it's not grouped by label
    for s in split:
        rng.shuffle(split[s])

    return split


def print_summary(split: dict):
    print(f"\n{'='*50}")
    print(f"  Split summary (seed=42, stratified)")
    print(f"{'='*50}")
    for name, clips in split.items():
        n_violent     = sum(1 for c in clips if c["label"] == 1)
        n_nonviolent  = sum(1 for c in clips if c["label"] == 0)
        print(f"\n  {name:5s}: {len(clips):4d} clips  "
              f"(violent={n_violent}, non-violent={n_nonviolent})")
    total = sum(len(v) for v in split.values())
    print(f"\n  Total: {total} clips")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",  type=str, default="./RWF-2000")
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.15)
    args = parser.parse_args()

    root = Path(args.root)
    print(f"\n  Collecting clips from {root}...")
    clips_by_label = collect_clips(root)

    for label, clips in clips_by_label.items():
        print(f"  {label}: {len(clips)} clips found")

    split = stratified_split(
        clips_by_label,
        train_ratio=args.train,
        val_ratio=args.val,
        seed=args.seed,
    )

    print_summary(split)

    # add metadata to the file
    output = {
        "meta": {
            "dataset":     "RWF-2000",
            "seed":        args.seed,
            "train_ratio": args.train,
            "val_ratio":   args.val,
            "test_ratio":  round(1.0 - args.train - args.val, 2),
            "label_map":   LABEL_MAP,
        },
        "train": split["train"],
        "val":   split["val"],
        "test":  split["test"],
    }

    out = Path("split.json")
    with open(out, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved → {out}")
    print(f"  Use this file for all models M1, M2, M3, M4.\n")