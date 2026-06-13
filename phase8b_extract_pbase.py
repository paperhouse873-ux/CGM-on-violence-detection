"""
Phase 8b — Extract p_base for the NEW detectors (SlowFast / MViT)
==========================================================
After fine-tuning (phase8), we need the new detector's p_base over ALL 1989
clips to attach CGM (Stage C). Context streams (crowd/light/motion) do NOT
depend on the detector -> REUSE the old cache, only re-extract p_base.

EXTREMELY IMPORTANT — clip order:
  Must be EXACTLY like Phase 3: walk split.json as train -> val -> test.
  That way p_base_<model>.npy lines up with cache/labels.npy, splits.npy,
  z_crowd.npy, z_light.npy, z_motion.npy (same index).

Output:
  cache/p_base_<model>.npy        (shape (1989,), float32)

Test LOCALLY first (no real checkpoint needed, just test pipeline + order):
  python phase8b_extract_pbase.py --model slowfast_r50 --limit 6 --allow_pretrained
  python phase8b_extract_pbase.py --model mvit_base_16x4 --limit 6 --allow_pretrained

Real run on A100 (after the fine-tuned checkpoint exists):
  python phase8b_extract_pbase.py --model slowfast_r50
  python phase8b_extract_pbase.py --model mvit_base_16x4
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from phase1_dataset import KINETICS_MEAN, KINETICS_STD
from phase8_finetune_detectors import build_model, pack_input, N_FRAMES, DEVICE

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
CKPT = Path("checkpoints")


class ClipDataset(torch.utils.data.Dataset):
    """Read clip -> tensor (3, T, H, W). Same frame sampling as Phase 3."""
    def __init__(self, samples, root, n_frames, img_size=224):
        self.samples = samples
        self.root = Path(root)
        self.n_frames = n_frames
        self.img_size = img_size
        self.mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.root / self.samples[idx]["path"]
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.n_frames
        idxs = np.linspace(0, max(total - 1, 0), self.n_frames, dtype=int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, f = cap.read()
            if not ret or f is None:
                f = frames[-1].copy() if frames else \
                    np.zeros((self.img_size, self.img_size, 3), np.uint8)
            f = cv2.resize(f, (self.img_size, self.img_size))
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
        cap.release()
        t = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0
                             ).permute(3, 0, 1, 2)
        t = (t - self.mean) / self.std
        return t, idx


def build_ordered_samples(split_file):
    """Reproduce EXACTLY the Phase 3 order: train -> val -> test."""
    with open(split_file) as f:
        sd = json.load(f)
    samples, split_ids = [], []
    for sid, sname in enumerate(["train", "val", "test"]):
        for item in sd[sname]:
            samples.append(item)
            split_ids.append(sid)
    return samples, np.array(split_ids, dtype=np.int32)


@torch.no_grad()
def main(args):
    model_name = args.model
    nf = N_FRAMES[model_name]
    print("=" * 66)
    print(f"  Extract p_base — {model_name}  (n_frames={nf}, device={DEVICE})")
    print("=" * 66)

    # 1) clip order same as Phase 3
    samples, split_ids = build_ordered_samples(args.split)
    if args.limit:
        samples = samples[:args.limit]
        split_ids = split_ids[:args.limit]
    N = len(samples)
    print(f"  clips: {N}")

    # 2) build model + load fine-tuned checkpoint
    ckpt_path = CKPT / f"{model_name}_best.pth"
    model = build_model(model_name)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        print(f"  loaded checkpoint: {ckpt_path}  (val F1={ck.get('val_metrics',{}).get('f1','?')})")
    else:
        if not args.allow_pretrained:
            raise FileNotFoundError(
                f"Could not find {ckpt_path}. Fine-tune first (phase8), "
                f"or use --allow_pretrained to TEST the pipeline locally.")
        print(f"  [WARNING] no {ckpt_path} yet -> using pretrained weights "
              f"ONLY to test the pipeline (p_base will NOT be valid for the paper).")
    model.eval()

    # 3) extract p_base in order
    ds = ClipDataset(samples, args.root, nf)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers)
    probs = np.zeros(N, dtype=np.float32)
    for videos, idxs in tqdm(loader, desc="  [p_base]", unit="batch"):
        videos = videos.to(DEVICE)
        x = pack_input(videos, model_name)            # pack fresh each batch
        p = torch.sigmoid(model(x).squeeze(1)).cpu().numpy()
        for j, ix in zip(p, idxs.numpy()):
            probs[ix] = j

    # 4) save (only save the full version; --limit is for testing, don't overwrite)
    if args.limit:
        print(f"\n  [TEST MODE limit={args.limit}] p_base sample: "
              f"{np.round(probs[:args.limit], 3).tolist()}")
        print("  Pipeline + order OK. NOT saving file (test mode).")
    else:
        out = CACHE / f"p_base_{model_name}.npy"
        np.save(out, probs)
        # sanity: length must match labels cache
        if (CACHE / "labels.npy").exists():
            nlab = len(np.load(CACHE / "labels.npy"))
            ok = (nlab == N)
            print(f"\n  saved -> {out}  shape={probs.shape}")
            print(f"  matches labels.npy (N={nlab}): {'OK' if ok else 'MISMATCH!'}")
        print(f"  p_base mean={probs.mean():.4f}  min={probs.min():.4f}  max={probs.max():.4f}")
    print("  done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["slowfast_r50", "mvit_base_16x4"])
    ap.add_argument("--root", default="RWF-2000")
    ap.add_argument("--split", default="split.json")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N clips to TEST the pipeline locally.")
    ap.add_argument("--allow_pretrained", action="store_true",
                    help="Allow running without a checkpoint (test only).")
    args = ap.parse_args()
    main(args)
