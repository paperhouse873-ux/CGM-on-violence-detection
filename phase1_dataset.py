import cv2
import json
import time
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ── Kinetics-400 mean/std (the standard X3D uses) ──────────────────────
KINETICS_MEAN = [0.45, 0.45, 0.45]
KINETICS_STD  = [0.225, 0.225, 0.225]


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — Dataset class
# ═══════════════════════════════════════════════════════════════════════════

class RWF2000Dataset(Dataset):
    """
    RWF-2000 dataset for PyTorch.

    Input:  video path
    Output: tensor (C, T, H, W) = (3, 16, 224, 224), label (0 or 1)

    Frame sampling:
      take T=16 evenly spaced frames across the video.
      e.g. 150-frame video → frames [0, 9, 18, ..., 140, 149].
      if the video is shorter than T frames → repeat the last frame to fill.

    Args:
        root:       path to RWF-2000 root dir
        split_file: path to split.json (from split.py)
        split:      "train", "val", or "test"
        n_frames:   number of frames to sample (default 16 — X3D-S standard)
        img_size:   resize size (default 224)
        augment:    enable data augmentation for training
    """

    def __init__(
        self,
        root:       str | Path,
        split_file: str | Path,
        split:      str = "train",
        n_frames:   int = 16,
        img_size:   int = 224,
        augment:    bool = False,
    ):
        self.root      = Path(root)
        self.n_frames  = n_frames
        self.img_size  = img_size
        self.augment   = augment

        # Load split
        with open(split_file) as f:
            data = json.load(f)

        if split not in data:
            raise ValueError(f"Split '{split}' not in {split_file}. "
                             f"Available: {list(data.keys())}")

        self.samples = data[split]  # list of {"path": ..., "label": ...}
        self.split   = split

        print(f"  RWF2000Dataset [{split}]: {len(self.samples)} clips")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_frames(self, path: Path) -> np.ndarray:
        """
        Read the video and sample T frames evenly.
        Returns array (T, H, W, C) in BGR order (OpenCV).
        """
        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            # fallback: read everything then sample
            frames_raw = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_raw.append(frame)
            cap.release()
            total = len(frames_raw)
        else:
            frames_raw = None

        # file won't open (bad name / missing / codec error) -> return black frames
        # instead of crashing. keeps the pipeline running and dataset alignment intact.
        if total <= 0:
            return np.zeros((self.n_frames, self.img_size, self.img_size, 3),
                            dtype=np.uint8)

        # compute indices to read
        if total >= self.n_frames:
            indices = np.linspace(0, total - 1, self.n_frames, dtype=int)
        else:
            # video shorter than n_frames: use all, pad with last frame
            indices = list(range(total)) + \
                      [total - 1] * (self.n_frames - total)
            indices = np.array(indices)

        frames = []
        if frames_raw is not None:
            # already read everything earlier
            for idx in indices:
                idx = min(idx, len(frames_raw) - 1)
                frames.append(frames_raw[idx])
        else:
            # use seek
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    # reuse last frame read
                    frame = frames[-1] if frames else np.zeros(
                        (self.img_size, self.img_size, 3), dtype=np.uint8)
                frames.append(frame)
            cap.release()

        return np.stack(frames)  # (T, H_orig, W_orig, 3)

    def _preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """
        frames: (T, H, W, 3) BGR uint8
        → tensor (3, T, H, W) float32, normalized
        """
        processed = []
        for frame in frames:
            # resize
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # augmentation (train only)
            if self.augment:
                frame = self._augment(frame)
            # uint8 → float [0,1]
            frame = frame.astype(np.float32) / 255.0
            processed.append(frame)

        # stack: (T, H, W, 3) → (3, T, H, W)
        tensor = torch.from_numpy(np.stack(processed))  # (T, H, W, 3)
        tensor = tensor.permute(3, 0, 1, 2)             # (3, T, H, W)

        # normalize with Kinetics mean/std
        mean = torch.tensor(KINETICS_MEAN).view(3, 1, 1, 1)
        std  = torch.tensor(KINETICS_STD).view(3, 1, 1, 1)
        tensor = (tensor - mean) / std

        return tensor  # (3, 16, 224, 224)

    def _augment(self, frame: np.ndarray) -> np.ndarray:
        """simple augmentation: random horizontal flip"""
        if random.random() > 0.5:
            frame = cv2.flip(frame, 1)
        return frame

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        path   = self.root / sample["path"]
        label  = sample["label"]

        frames = self._sample_frames(path)
        tensor = self._preprocess(frames)

        return tensor, label


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — Test DataLoader
# ═══════════════════════════════════════════════════════════════════════════

def test_dataloader(root: str, split_file: str):
    print(f"\n{'='*55}")
    print(f"  Phase 1 — DataLoader Test")
    print(f"{'='*55}\n")

    # ── build datasets ───────────────────────────────────────────────────
    train_ds = RWF2000Dataset(
        root=root, split_file=split_file,
        split="train", augment=True
    )
    val_ds = RWF2000Dataset(
        root=root, split_file=split_file,
        split="val", augment=False
    )
    test_ds = RWF2000Dataset(
        root=root, split_file=split_file,
        split="test", augment=False
    )

    # ── DataLoader ───────────────────────────────────────────────────────
    train_loader = DataLoader(
    train_ds,
    batch_size=8,       # bump batch size to compensate
    shuffle=True,
    num_workers=0,      # Windows: must be 0
    pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, num_workers=2
    )

    # ── test 1 batch ─────────────────────────────────────────────────────
    print(f"  Testing 1 batch from train_loader...")
    start = time.time()
    videos, labels = next(iter(train_loader))
    elapsed = time.time() - start

    print(f"\n  [Shape check]")
    print(f"    videos.shape : {tuple(videos.shape)}")
    print(f"    Expected     : (4, 3, 16, 224, 224)")
    shape_ok = tuple(videos.shape) == (4, 3, 16, 224, 224)
    print(f"    Status       : {'PASS' if shape_ok else 'FAIL'}")

    print(f"\n  [Label check]")
    print(f"    labels       : {labels.tolist()}")
    print(f"    Unique values: {labels.unique().tolist()}")
    label_ok = set(labels.tolist()).issubset({0, 1})
    print(f"    Status       : {'PASS' if label_ok else 'FAIL'}")

    print(f"\n  [Normalization check]")
    print(f"    Mean: {videos.mean():.4f}  (should be near 0)")
    print(f"    Std : {videos.std():.4f}   (should be near 1)")

    print(f"\n  [Speed test] — 10 batches")
    start = time.time()
    n_clips = 0
    for i, (v, l) in enumerate(train_loader):
        n_clips += v.shape[0]
        if i >= 9:
            break
    elapsed_10 = time.time() - start
    speed = n_clips / elapsed_10

    print(f"    {n_clips} clips in {elapsed_10:.2f}s")
    print(f"    Speed: {speed:.1f} clips/s")
    speed_ok = speed >= 5  # minimum threshold 5 clips/s
    print(f"    Status: {'PASS' if speed_ok else 'SLOW — lower num_workers or raise batch_size'}")

    # ── summary ──────────────────────────────────────────────────────────
    all_pass = shape_ok and label_ok and speed_ok
    print(f"\n{'─'*55}")
    print(f"  Overall: {'ALL PASS — DataLoader ready' if all_pass else 'Issues — see details above'}")
    print(f"{'='*55}\n")

    if all_pass:
        print("  DataLoader ready for Phase 2 (Fine-tune X3D-S).\n")

    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",  type=str,
                        default="./RWF-2000",
                        help="path to RWF-2000 root dir")
    parser.add_argument("--split", type=str,
                        default="./split.json",
                        help="path to split.json from split.py")
    args = parser.parse_args()

    test_dataloader(args.root, args.split)