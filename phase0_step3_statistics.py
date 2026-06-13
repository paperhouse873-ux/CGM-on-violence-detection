import cv2
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}


def get_video_info(path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    fps         = cap.get(cv2.CAP_PROP_FPS)
    n_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration    = n_frames / fps if fps > 0 else 0
    cap.release()

    return {
        "path":       str(path),
        "split":      path.parts[-3],    # train / val
        "label":      path.parts[-2],    # fight / nonFight
        "filename":   path.name,
        "fps":        round(fps, 2),
        "n_frames":   n_frames,
        "duration_s": round(duration, 3),
        "width":      width,
        "height":     height,
        "resolution": f"{width}x{height}",
    }


def collect_all_videos(root: Path) -> list[Path]:
    files = []
    for folder in root.rglob("*"):
        if folder.is_dir():
            for f in folder.iterdir():
                if f.suffix.lower() in VIDEO_EXTS:
                    files.append(f)
    return sorted(files)


def print_stats(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print(f"  RWF-2000 Dataset Statistics")
    print(f"{'='*60}")

    # Class distribution
    print(f"\n  Class distribution:")
    dist = df.groupby(["split", "label"]).size().reset_index(name="count")
    for _, row in dist.iterrows():
        bar = "█" * (row["count"] // 50)
        print(f"    {row['split']:5s}/{row['label']:8s}: {row['count']:4d}  {bar}")

    # Duration
    print(f"\n  Duration (seconds):")
    print(f"    Mean:   {df['duration_s'].mean():.2f}s")
    print(f"    Std:    {df['duration_s'].std():.2f}s")
    print(f"    Min:    {df['duration_s'].min():.2f}s")
    print(f"    Max:    {df['duration_s'].max():.2f}s")
    print(f"    Median: {df['duration_s'].median():.2f}s")

    # FPS
    fps_counts = df["fps"].value_counts().head(5)
    print(f"\n  FPS distribution (top 5):")
    for fps, cnt in fps_counts.items():
        print(f"    {fps:6.2f} fps: {cnt} clips")

    # Resolution
    res_counts = df["resolution"].value_counts().head(5)
    print(f"\n  Resolution (top 5):")
    for res, cnt in res_counts.items():
        print(f"    {res:12s}: {cnt} clips")

    # Frame count
    print(f"\n  Frame count per clip:")
    print(f"    Mean:   {df['n_frames'].mean():.1f}")
    print(f"    Std:    {df['n_frames'].std():.1f}")
    print(f"    Min:    {df['n_frames'].min()}")
    print(f"    Max:    {df['n_frames'].max()}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./RWF-2000")
    args = parser.parse_args()
    root = Path(args.root)

    files = collect_all_videos(root)
    print(f"\n  Found {len(files)} video files. Extracting stats...")

    records = []
    for f in tqdm(files, desc="  Scanning", unit="file"):
        info = get_video_info(f)
        if info:
            records.append(info)

    df = pd.DataFrame(records)

    # print stats
    print_stats(df)

    # save CSV
    out = Path("dataset_stats.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")
    print(f"  Columns: {list(df.columns)}\n")