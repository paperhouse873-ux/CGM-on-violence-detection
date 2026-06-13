import cv2
import json
import pickle
import torch
import argparse
import numpy as np
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).parent))
from phase1_dataset import RWF2000Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════
# STREAM 1 — p_base from X3D-S (frozen)
# ═══════════════════════════════════════════════════════════════════════════

def load_frozen_x3ds(ckpt_path: str) -> nn.Module:
    print("\n  Loading frozen X3D-S checkpoint...")
    from pytorchvideo.models.hub import x3d_s
    model = x3d_s(pretrained=False)
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, 1)
    if hasattr(model.blocks[-1], "act"):
        model.blocks[-1].act = nn.Identity()
    if hasattr(model.blocks[-1], "activation"):
        model.blocks[-1].activation = nn.Identity()

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Checkpoint loaded from epoch {ckpt.get('epoch', '?')}")
    return model


@torch.no_grad()
def extract_p_base(model, dataset, batch_size=8) -> np.ndarray:
    """Extract p_base from X3D-S for the whole dataset."""
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_probs = []
    for videos, _ in tqdm(loader, desc="  [p_base]", unit="batch"):
        videos = videos.to(DEVICE)
        logits = model(videos).squeeze(1)
        probs  = torch.sigmoid(logits)
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# STREAM 2 — Crowd density (YOLOv8n)
# ═══════════════════════════════════════════════════════════════════════════

def extract_crowd_features(video_path: str,
                           sample_every: int = 4) -> np.ndarray:
    """
    Run YOLOv8n every `sample_every` frames, count class=person.
    Returns z_crowd (4,): [mean_count, max_count, count_variance, density_area]
    """
    from ultralytics import YOLO
    if not hasattr(extract_crowd_features, "_model"):
        extract_crowd_features._model = YOLO("yolov8n.pt")
    yolo = extract_crowd_features._model

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    counts = []
    areas  = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            h, w = frame.shape[:2]
            results = yolo(frame, classes=[0], verbose=False)[0]
            n_persons = len(results.boxes)
            counts.append(n_persons)
            if n_persons > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                box_areas = ((boxes[:, 2] - boxes[:, 0]) *
                             (boxes[:, 3] - boxes[:, 1]))
                total_area = box_areas.sum() / (h * w)
            else:
                total_area = 0.0
            areas.append(total_area)
        frame_idx += 1
    cap.release()

    if not counts:
        return np.zeros(4, dtype=np.float32)

    counts = np.array(counts, dtype=np.float32)
    return np.array([
        counts.mean(),
        counts.max(),
        counts.var(),
        np.mean(areas),
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# STREAM 3 — Lighting condition (OpenCV)
# ═══════════════════════════════════════════════════════════════════════════

def extract_lighting_features(video_path: str) -> np.ndarray:
    """
    Compute lighting features with OpenCV.
    Returns z_light (4,): [mean_brightness, contrast_std, blur_score, low_light_ratio]
    """
    cap = cv2.VideoCapture(video_path)
    brightness_list, contrast_list, blur_list, dark_list = [], [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brightness & contrast
        mean, std = cv2.meanStdDev(gray)
        brightness_list.append(float(mean[0][0]))
        contrast_list.append(float(std[0][0]))

        # blur score (Laplacian variance — low = blurry)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_list.append(blur)

        # low light ratio (fraction of dark pixels below threshold 50/255)
        dark_ratio = np.mean(gray < 50)
        dark_list.append(dark_ratio)

    cap.release()

    if not brightness_list:
        return np.zeros(4, dtype=np.float32)

    return np.array([
        np.mean(brightness_list) / 255.0,   # normalize [0,1]
        np.mean(contrast_list)   / 128.0,   # normalize
        np.mean(blur_list)       / 1000.0,  # normalize
        np.mean(dark_list),                 # already [0,1]
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# STREAM 4 — Motion + Synchrony (Farneback optical flow)
# ═══════════════════════════════════════════════════════════════════════════

def extract_motion_features(video_path: str) -> np.ndarray:
    """
    Use Farneback optical flow.
    Returns z_motion (4,):
      [motion_mean, motion_peak, direction_entropy, motion_synchrony★]

    motion_synchrony: 1 = everything moving the same way (dancing)
                      0 = chaotic motion (violence)
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.zeros(4, dtype=np.float32)

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    magnitudes, entropies, synchronies = [], [], []

    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        magnitudes.append(float(mag.mean()))

        # direction entropy — high = chaotic
        hist, _ = np.histogram(ang.flatten(), bins=8, range=(0, 2*np.pi))
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist * np.log(hist + 1e-8))
        entropies.append(entropy)

        # motion synchrony — circular variance of direction
        # synchrony = 1 - circular_variance
        # circular_variance = 1 - |mean(exp(i*theta))|
        theta = ang.flatten()
        r = np.abs(np.mean(np.exp(1j * theta)))  # [0,1]
        synchronies.append(float(r))             # 1=synchronized, 0=chaotic

        prev_gray = curr_gray

    cap.release()

    if not magnitudes:
        return np.zeros(4, dtype=np.float32)

    mags = np.array(magnitudes)
    return np.array([
        mags.mean(),                    # motion_mean
        mags.max(),                     # motion_peak
        np.mean(entropies),             # direction_entropy
        np.mean(synchronies),           # motion_synchrony ★
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — run the full extraction pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_extraction(args):
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    # Load split
    with open(args.split) as f:
        split_data = json.load(f)

    # merge all samples with their split label
    all_samples = []
    split_ids   = []
    for sid, sname in enumerate(["train", "val", "test"]):
        for item in split_data[sname]:
            all_samples.append(item)
            split_ids.append(sid)

    N = len(all_samples)
    root = Path(args.root)
    print(f"\n  Total clips to process: {N}")

    # ── STREAM 1: p_base from X3D-S ───────────────────────────────────
    p_base_path = cache_dir / "p_base.npy"
    if p_base_path.exists() and not args.force:
        print(f"\n  [SKIP] p_base already cached.")
        p_base = np.load(p_base_path)
    else:
        print(f"\n  Extracting p_base (X3D-S)...")

        class SimpleDS(torch.utils.data.Dataset):
            def __init__(self, samples, root, n_frames=16, img_size=224):
                self.samples  = samples
                self.root     = Path(root)
                self.n_frames = n_frames
                self.img_size = img_size
                from phase1_dataset import KINETICS_MEAN, KINETICS_STD
                self.mean = torch.tensor(KINETICS_MEAN).view(3,1,1,1)
                self.std  = torch.tensor(KINETICS_STD).view(3,1,1,1)

            def __len__(self): return len(self.samples)

            def __getitem__(self, idx):
                path = self.root / self.samples[idx]["path"]
                cap  = cv2.VideoCapture(str(path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 16
                idxs  = np.linspace(0, total-1, self.n_frames, dtype=int)
                frames = []
                for i in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                    ret, f = cap.read()
                    if not ret or f is None:
                        f = frames[-1] if frames else np.zeros(
                            (self.img_size, self.img_size, 3), np.uint8)
                    f = cv2.resize(f, (self.img_size, self.img_size))
                    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                    frames.append(f)
                cap.release()
                t = torch.from_numpy(
                    np.stack(frames).astype(np.float32) / 255.0
                ).permute(3,0,1,2)
                t = (t - self.mean) / self.std
                return t, self.samples[idx]["label"]

        model = load_frozen_x3ds(args.ckpt)
        ds = SimpleDS(all_samples, root)
        p_base = extract_p_base(model, ds, batch_size=8)
        np.save(p_base_path, p_base)
        print(f"  Saved → {p_base_path}  shape={p_base.shape}")

    # ── STREAM 2: z_crowd ─────────────────────────────────────────────
    z_crowd_path = cache_dir / "z_crowd.npy"
    if z_crowd_path.exists() and not args.force:
        print(f"\n  [SKIP] z_crowd already cached.")
        z_crowd = np.load(z_crowd_path)
    else:
        print(f"\n  Extracting z_crowd (YOLOv8n)...")
        z_crowd = np.zeros((N, 4), dtype=np.float32)
        for i, item in enumerate(tqdm(all_samples, desc="  [crowd]")):
            path = str(root / item["path"])
            z_crowd[i] = extract_crowd_features(path)
        np.save(z_crowd_path, z_crowd)
        print(f"  Saved → {z_crowd_path}  shape={z_crowd.shape}")

    # ── STREAM 3: z_light ─────────────────────────────────────────────
    z_light_path = cache_dir / "z_light.npy"
    if z_light_path.exists() and not args.force:
        print(f"\n  [SKIP] z_light already cached.")
        z_light = np.load(z_light_path)
    else:
        print(f"\n  Extracting z_light (OpenCV)...")
        z_light = np.zeros((N, 4), dtype=np.float32)
        for i, item in enumerate(tqdm(all_samples, desc="  [light]")):
            path = str(root / item["path"])
            z_light[i] = extract_lighting_features(path)
        np.save(z_light_path, z_light)
        print(f"  Saved → {z_light_path}  shape={z_light.shape}")

    # ── STREAM 4: z_motion ────────────────────────────────────────────
    z_motion_path = cache_dir / "z_motion.npy"
    if z_motion_path.exists() and not args.force:
        print(f"\n  [SKIP] z_motion already cached.")
        z_motion = np.load(z_motion_path)
    else:
        print(f"\n  Extracting z_motion (Farneback)...")
        z_motion = np.zeros((N, 4), dtype=np.float32)
        for i, item in enumerate(tqdm(all_samples, desc="  [motion]")):
            path = str(root / item["path"])
            z_motion[i] = extract_motion_features(path)
        np.save(z_motion_path, z_motion)
        print(f"  Saved → {z_motion_path}  shape={z_motion.shape}")

    # ── Labels & splits ───────────────────────────────────────────────
    labels = np.array([s["label"] for s in all_samples], dtype=np.int32)
    splits = np.array(split_ids, dtype=np.int32)
    np.save(cache_dir / "labels.npy", labels)
    np.save(cache_dir / "splits.npy", splits)

    # ── Concat + Standardize → 13-dim ─────────────────────────────────
    print(f"\n  Building 13-dim context vector...")
    X_raw = np.concatenate([
        p_base.reshape(-1, 1),  # 1-dim
        z_crowd,                # 4-dim
        z_light,                # 4-dim
        z_motion,               # 4-dim
    ], axis=1)                  # → (N, 13)

    # Fit StandardScaler on the train set ONLY
    train_mask = splits == 0
    scaler = StandardScaler()
    scaler.fit(X_raw[train_mask])
    X_norm = scaler.transform(X_raw).astype(np.float32)

    np.save(cache_dir / "context_13dim.npy", X_norm)
    with open(cache_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # ── Sanity check ──────────────────────────────────────────────────
    print(f"\n  {'='*55}")
    print(f"  Extraction complete — Sanity check")
    print(f"  {'='*55}")
    print(f"  N total      : {N}")
    print(f"  Train / Val / Test : {train_mask.sum()} / "
          f"{(splits==1).sum()} / {(splits==2).sum()}")
    print(f"  p_base  mean : {p_base.mean():.4f}  std: {p_base.std():.4f}")
    print(f"  p_base violent clips : {p_base[labels==1].mean():.4f}")
    print(f"  p_base normal  clips : {p_base[labels==0].mean():.4f}")

    print(f"\n  motion_synchrony check (★ key feature):")
    sync_col = z_motion[:, 3]
    print(f"  Violent clips  synchrony: {sync_col[labels==1].mean():.4f}")
    print(f"  Normal  clips  synchrony: {sync_col[labels==0].mean():.4f}")
    if sync_col[labels==0].mean() > sync_col[labels==1].mean():
        print(f"  PASS — Normal > Violent (dancing is more synchronized than violence)")
    else:
        print(f"  NOTE — Unexpected direction, check feature logic")

    print(f"\n  context_13dim shape : {X_norm.shape}")
    print(f"  Saved all to cache/")
    print(f"\n  Phase 3 DONE. Next: Phase 4 — Train Context Gating Module.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",  type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--ckpt",  type=str,
                        default="checkpoints/x3ds_best.pth")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract everything even if cached")
    args = parser.parse_args()
    run_extraction(args)