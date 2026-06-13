"""
Phase 15 — Latency / cost breakdown (per clip)
==============================================
Measure real time (ms/clip) of each component to close the "lightweight" story:
the three context streams (crowd/light/motion), the CGM forward, and the X3D-S
detector forward for reference. All on CPU (current environment), averaged over
N test clips.

Output:
  results/phase15_latency.json

Run:  python phase15_latency.py --root RWF-2000 --n 15
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent))
from phase3_extract_context import (
    extract_crowd_features, extract_lighting_features, extract_motion_features,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE = Path("cache"); CKPT = Path("checkpoints")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)


class CGM(nn.Module):
    def __init__(self, d=13, w=32):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d, w), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(w, 1), nn.Sigmoid())
        self.ctx = nn.Sequential(nn.Linear(d, w), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(w, 1), nn.Sigmoid())

    def forward(self, x, pb):
        a = self.gate(x).squeeze(1); c = self.ctx(x).squeeze(1)
        return a * pb + (1 - a) * c, a, c


def load_test_paths(n):
    sj = json.load(open("split.json", encoding="utf-8"))
    paths = []
    for key in ("train", "val", "test"):
        for it in sj[key]:
            paths.append(it["path"])
    splits = np.load(CACHE / "splits.npy")
    test_idx = np.where(splits == 2)[0]
    rng = np.random.default_rng(0)
    sel = rng.choice(test_idx, size=min(n, len(test_idx)), replace=False)
    return [paths[i] for i in sel]


def timeit(fn, *a, **k):
    t0 = time.perf_counter(); fn(*a, **k); return (time.perf_counter() - t0) * 1000.0


def main(args):
    root = Path(args.root)
    paths = load_test_paths(args.n)
    print("=" * 64)
    print(f"PHASE 15 — Latency (CPU), N={len(paths)} test clips")
    print("=" * 64)

    # warm-up YOLO (model load not counted)
    _ = extract_crowd_features(str(root / paths[0]))

    streams = {"crowd (YOLOv8n)": [], "lighting (OpenCV)": [],
               "motion (Farneback)": []}
    for p in paths:
        fp = str(root / p)
        streams["crowd (YOLOv8n)"].append(timeit(extract_crowd_features, fp))
        streams["lighting (OpenCV)"].append(timeit(extract_lighting_features, fp))
        streams["motion (Farneback)"].append(timeit(extract_motion_features, fp))

    # CGM forward (per clip; time a single-row forward averaged)
    model = CGM(); model.eval()
    x = torch.randn(1, 13); pb = torch.rand(1)
    cgm_t = []
    with torch.no_grad():
        for _ in range(200):
            cgm_t.append(timeit(lambda: model(x, pb)))
    cgm_ms = float(np.mean(cgm_t))

    # X3D-S forward (reference detector cost) on a 16-frame clip tensor
    x3d_ms = None
    try:
        from pytorchvideo.models.hub import x3d_s
        det = x3d_s(pretrained=False)
        inf = det.blocks[-1].proj.in_features
        det.blocks[-1].proj = nn.Linear(inf, 1); det.eval()
        clip = torch.randn(1, 3, 16, 224, 224)
        with torch.no_grad():
            _ = det(clip)  # warm-up
            xs = [timeit(lambda: det(clip)) for _ in range(5)]
        x3d_ms = float(np.mean(xs))
    except Exception as e:
        print(f"  [warn] X3D-S timing skipped: {e}")

    def stat(v):
        v = np.array(v, float); return float(v.mean()), float(v.std())

    out = {"n_clips": len(paths), "device": "cpu", "streams_ms": {}}
    print(f"\n  {'component':24} {'ms/clip':>12}")
    ctx_total = 0.0
    for k, v in streams.items():
        m, s = stat(v); ctx_total += m
        out["streams_ms"][k] = [m, s]
        print(f"  {k:24} {m:8.1f} ± {s:5.1f}")
    out["context_total_ms"] = ctx_total
    out["cgm_ms"] = cgm_ms
    out["x3ds_ms"] = x3d_ms
    print(f"  {'context total':24} {ctx_total:8.1f}")
    print(f"  {'CGM forward':24} {cgm_ms:8.3f}")
    if x3d_ms:
        print(f"  {'X3D-S forward (ref)':24} {x3d_ms:8.1f}")

    json.dump(out, open(RESULTS / "phase15_latency.json", "w"), indent=2)
    print(f"\n  saved -> results/phase15_latency.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="RWF-2000")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()
    main(args)
