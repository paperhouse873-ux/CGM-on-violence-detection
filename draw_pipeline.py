"""
Draw the full project pipeline (Phase 0 -> submission) as one 300 DPI figure.
Color by status: Done (teal) · In Progress (orange) · Pending (gray).

Run:
  "C:/Users/HA VIET HUNG/.conda/envs/violence_det/python.exe" draw_pipeline.py
Output:
  figures/PROJECT_PIPELINE.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

FIGDIR = Path("figures"); FIGDIR.mkdir(exist_ok=True)

# ── Palette (matches the pptx slide) ────────────────────────────────────────
NAVY   = "#0F1F3D"
TEAL   = "#0D9488"
PURPLE = "#7C3AED"
ORANGE = "#D97706"
GRAY   = "#64748B"
INK    = "#1F2937"
LIGHT  = "#F1F5F9"

DONE, PROG, PEND = TEAL, ORANGE, GRAY

fig, ax = plt.subplots(figsize=(16, 9.2))
ax.set_xlim(0, 16); ax.set_ylim(0, 9.2); ax.axis("off")

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(8, 8.9, "Context-Aware Violence Detection — Full Project Pipeline",
        ha="center", va="center", fontsize=19, fontweight="bold", color=NAVY)
ax.text(8, 8.5, "Two-stage, model-agnostic framework for false-alarm reduction "
        "(X3D-S + Context Gating Module)",
        ha="center", va="center", fontsize=11, color=GRAY, style="italic")


def stage(x, y, w, h, color, num, title, lines, fc=None, text_light=True):
    """Draw one rounded stage block."""
    fc = fc if fc else color
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                         linewidth=2, edgecolor=color, facecolor=fc, alpha=1.0,
                         mutation_aspect=1)
    ax.add_patch(box)
    # round number badge
    ax.add_patch(plt.Circle((x + 0.42, y + h - 0.42), 0.27, color=color, zorder=5))
    ax.text(x + 0.42, y + h - 0.42, num, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", zorder=6)
    tcol = "white" if text_light else INK
    ax.text(x + 0.85, y + h - 0.40, title, ha="left", va="center",
            fontsize=11.5, fontweight="bold", color=tcol)
    for i, ln in enumerate(lines):
        ax.text(x + 0.25, y + h - 0.95 - i * 0.40, ln, ha="left", va="center",
                fontsize=9.2, color=tcol)


def arrow(x1, y1, x2, y2, color=NAVY, style="-|>", lw=2.2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle=style, mutation_scale=18, linewidth=lw,
                 color=color, connectionstyle=f"arc3,rad={rad}", zorder=3))


# light fill for dark blocks: use the strong color + white text
def lite(hexcol):
    return hexcol  # keep strong, white text

# ============================================================================
# ROW 1 — Data prep & stage 1 (Done)
# ============================================================================
y1 = 6.0
stage(0.4,  y1, 3.4, 1.9, DONE, "0",
      "Data Preparation",
      ["RWF-2000: 1,989 clips", "Check structure / integrity",
       "Stratified split 70-15-15", "-> split.json (seed 42)"])
stage(4.2,  y1, 3.4, 1.9, DONE, "1",
      "Dataset & Loader",
      ["RWF2000Dataset (PyTorch)", "Sample T=16 frames",
       "224x224, Kinetics norm", "DataLoader (num_workers=0)"])
stage(8.0,  y1, 3.4, 1.9, DONE, "2",
      "Stage-1: X3D-S",
      ["Fine-tune on RWF-2000", "Frozen after training",
       "BCEWithLogitsLoss", "-> p_base  (val F1=0.898)"])
stage(11.8, y1, 3.8, 1.9, DONE, "3",
      "Context Extraction",
      ["Crowd  (YOLOv8n) x4", "Lighting (OpenCV) x4",
       "Motion (Farneback) x4", "-> 13-dim vector + scaler"])

# row 1 arrows
arrow(3.8, y1+0.95, 4.2, y1+0.95)
arrow(7.6, y1+0.95, 8.0, y1+0.95)
arrow(11.4, y1+0.95, 11.8, y1+0.95)

# ============================================================================
# ROW 2 — Model & analysis (Done)
# arrow down from stage 3
# ============================================================================
y2 = 3.4
arrow(13.7, y1, 13.7, y2+1.9, rad=0.0)  # 3 -> 4 (going down)

stage(11.8, y2, 3.8, 1.9, DONE, "4",
      "Stage-2: CGM + Ablation",
      ["962-param gating module", "p_final = a.p_base+(1-a).p_ctx",
       "Ablation E0-E5", "-> FPR 0.153->0.113 (-26%)"])
stage(8.0,  y2, 3.4, 1.9, DONE, "4b",
      "EDA & Visualization",
      ["RQ1: FPR reduction", "RQ2: stream contribution",
       "RQ3: domain shift", "RQ4: motion redundancy"])
arrow(11.8, y2+0.95, 11.4, y2+0.95)  # 4 -> 4b (leftward)

# ============================================================================
# ROW 2 cont. — Cross-dataset (In Progress) & Paper (Pending)
# ============================================================================
stage(4.2,  y2, 3.4, 1.9, PROG, "5-6",
      "Cross-dataset (RLVS)",
      ["Zero-shot transfer", "RWF-2000 -> RLVS",
       "Re-measure FPR before/after", "-> generalization proof"])
arrow(8.0, y2+0.95, 7.6, y2+0.95)  # 4b -> 5/6 (leftward)

stage(0.4,  y2, 3.4, 1.9, PEND, "7",
      "Paper & Submission",
      ["Methodology / Experiments", "Results & Discussion",
       "Springer sn-jnl format", "-> EIDT (Scopus Q4)"])
arrow(4.2, y2+0.95, 3.8, y2+0.95)  # 5/6 -> 7 (leftward)

# ============================================================================
# STAGE 2 DETAIL BLOCK (CGM zoom) — below
# ============================================================================
yb = 0.5
detail = FancyBboxPatch((0.4, yb), 15.2, 2.3,
                        boxstyle="round,pad=0.02,rounding_size=0.10",
                        linewidth=1.4, edgecolor=NAVY, facecolor=LIGHT)
ax.add_patch(detail)
ax.text(0.7, yb + 2.02, "Inference detail — Context Gating Module (Stage 2)",
        ha="left", va="center", fontsize=11, fontweight="bold", color=NAVY)


def mini(x, y, w, h, color, title, sub, text_light=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.6, edgecolor=color, facecolor="white")
    ax.add_patch(b)
    ax.add_patch(plt.Rectangle((x, y), 0.10, h, color=color))
    ax.text(x + w/2, y + h - 0.30, title, ha="center", va="center",
            fontsize=9.6, fontweight="bold", color=INK)
    ax.text(x + w/2, y + 0.30, sub, ha="center", va="center",
            fontsize=8.3, color=GRAY)


yd = yb + 0.55
mini(0.8,  yd, 2.5, 1.05, TEAL,   "p_base", "X3D-S probability")
mini(3.7,  yd, 2.7, 1.05, PURPLE, "13-dim context", "crowd|light|motion")
mini(6.8,  yd, 2.7, 1.05, ORANGE, "MLP-gate -> a", "trust in X3D-S")
mini(9.9,  yd, 2.7, 1.05, ORANGE, "MLP-ctx -> p_ctx", "context correction")
mini(13.0, yd, 2.4, 1.05, NAVY,   "p_final", "a.p_base+(1-a).p_ctx")

arrow(3.3, yd+0.52, 3.7, yd+0.52, lw=1.8)
arrow(6.4, yd+0.52, 6.8, yd+0.52, lw=1.8)
arrow(9.5, yd+0.52, 9.9, yd+0.52, lw=1.8)
arrow(12.6, yd+0.52, 13.0, yd+0.52, lw=1.8)

# ── Status legend ────────────────────────────────────────────────────────────
handles = [
    Line2D([0],[0], marker="s", color="w", markerfacecolor=DONE, markersize=14,
           label="Done"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=PROG, markersize=14,
           label="In Progress"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=PEND, markersize=14,
           label="Pending"),
]
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.93),
          fontsize=10, frameon=True, ncol=3, columnspacing=1.0,
          handletextpad=0.4)

fig.savefig(FIGDIR / "PROJECT_PIPELINE.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved ->", FIGDIR / "PROJECT_PIPELINE.png")
