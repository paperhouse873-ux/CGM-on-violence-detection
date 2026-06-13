"""
DETAILED diagram of the Context Gating Module (CGM) — simple, easy to read, for the advisor.
Lays out: input 13-dim -> 2 MLP branches (gate / ctx) layer by layer -> alpha, p_ctx
-> fusion p_final = alpha*p_base + (1-alpha)*p_ctx.

Output: figures/FIG_CGM_detail.png  (300 DPI)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, MUTED = "#1b2430", "#5f6b7a"
PB    = "#3f6fb0"   # p_base
CROWD = "#2f8f7f"; LIGHTC = "#c08a2d"; MOTION = "#b5563f"
GATE  = "#6a4fb0"; CTX = "#2f7d9a"; FUSE = "#2d3b55"
PAPER, SOFT = "#ffffff", "#f4f6f9"
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")
ax.text(6.5, 5.92, "Context Gating Module (962 trainable parameters)",
        ha="center", va="center", fontsize=13, fontweight="bold", color=INK)


def rbox(x, y, w, h, ec, fc=PAPER, lw=1.7, r=0.10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.015,rounding_size={r}",
                 linewidth=lw, edgecolor=ec, facecolor=fc))


def arrow(x1, y1, x2, y2, color=INK, lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=13, linewidth=lw, color=color,
                 connectionstyle=f"arc3,rad={rad}", zorder=2))


def ctext(x, y, s, fs=9, color=INK, w="normal"):
    ax.text(x, y, s, ha="center", va="center", fontsize=fs, color=color, fontweight=w)


def layer(x, y, label, ec=MUTED, w=1.15, h=0.72):
    rbox(x, y, w, h, ec, PAPER, 1.5)
    ctext(x + w/2, y + h/2, label, 8.3, INK)
    return x + w  # right edge

# ── INPUT vector 13-dim (stacked colored cells) ──────────────────────────────
ix, iy, iw = 0.35, 2.55, 1.65
rbox(ix, iy, iw, 2.05, INK, SOFT, 1.8)
ctext(ix + iw/2, iy + 2.28, "Input  x", 9.6, INK, "bold")
ctext(ix + iw/2, iy + 2.04, r"$\in\mathbb{R}^{13}$", 8.6, MUTED)
cells = [("p_base (1)", PB), ("z_crowd (4)", CROWD),
         ("z_light (4)", LIGHTC), ("z_motion (4)", MOTION)]
ch = 0.45
for i, (lab, col) in enumerate(cells):
    yy = iy + 0.12 + (3 - i) * ch
    rbox(ix + 0.16, yy, iw - 0.32, ch - 0.08, col, PAPER, 1.3, r=0.05)
    ax.add_patch(plt.Rectangle((ix + 0.16, yy), 0.09, ch - 0.08, color=col))
    ctext(ix + iw/2 + 0.04, yy + (ch - 0.08)/2, lab, 7.7, INK)

# ── standardize ──────────────────────────────────────────────────────────────
sx = 2.55
rbox(sx, 3.05, 1.05, 1.05, MUTED, SOFT, 1.6)
ctext(sx + 0.525, 3.78, "standard-", 8.0, INK)
ctext(sx + 0.525, 3.55, "ise", 8.0, INK)
ctext(sx + 0.525, 3.28, "(train", 7.2, MUTED)
ctext(sx + 0.525, 3.10, "stats)", 7.2, MUTED)
arrow(ix + iw, 3.57, sx, 3.57)

# split point
split_x = sx + 1.05
arrow(split_x, 3.57, split_x + 0.35, 4.55, rad=-0.2)   # to gate lane
arrow(split_x, 3.57, split_x + 0.35, 2.05, rad=0.2)    # to ctx lane

# ── GATE lane (top) ──────────────────────────────────────────────────────────
def lane(y, ec, head, out_label, out_color):
    x = 4.05
    ax.text(x - 0.05, y + 0.95, head, ha="left", va="center", fontsize=9.2,
            fontweight="bold", color=ec)
    chain = ["Linear\n13\u219232", "ReLU", "Dropout\n0.3", "Linear\n32\u21921", "Sigmoid"]
    for lab in chain:
        xr = layer(x, y, lab, ec)
        arrow(xr, y + 0.36, xr + 0.16, y + 0.36, lw=1.4)
        x = xr + 0.16
    # output node
    rbox(x, y + 0.02, 0.95, 0.68, out_color, "#ffffff", 1.8)
    ctext(x + 0.475, y + 0.36, out_label, 11, out_color, "bold")
    return x + 0.95, y + 0.36

gx, gy = lane(4.30, GATE, "MLP-gate", r"$\alpha$", GATE)
cx, cy = lane(1.30, CTX, "MLP-ctx", r"$p_{ctx}$", CTX)

# ── FUSION ───────────────────────────────────────────────────────────────────
fx = 11.45
rbox(fx, 2.55, 1.25, 1.15, FUSE, "#eaeef5", 2.0)
ctext(fx + 0.625, 3.45, "fusion", 9.0, INK, "bold")
ctext(fx + 0.625, 3.12, r"$p_{final}=$", 8.6, INK)
ctext(fx + 0.625, 2.86, r"$\alpha\,p_{base}$", 7.8, INK)
ctext(fx + 0.625, 2.64, r"$+(1{-}\alpha)\,p_{ctx}$", 7.4, INK)

arrow(gx, gy, fx, 3.45, rad=0.0)          # alpha -> fusion
arrow(cx, cy, fx, 2.85, rad=0.0)          # p_ctx -> fusion

# p_base bypass to fusion (dashed)
ax.add_patch(FancyArrowPatch((ix + iw/2, iy + 0.34), (fx + 0.2, 2.55),
             arrowstyle="-|>", mutation_scale=13, linewidth=1.6,
             color=PB, linestyle=(0, (5, 3)),
             connectionstyle="arc3,rad=-0.32", zorder=1))
ax.text(7.0, 0.55, r"$p_{base}$ bypass", ha="center", fontsize=8.2, color=PB,
        style="italic")

# legend note
ax.text(6.5, 5.55, r"$\alpha$ = trust in detector   ·   "
        r"$\alpha\!\to\!1$: rely on $p_{base}$   ·   $\alpha\!\to\!0$: rely on context",
        ha="center", fontsize=8.6, color=MUTED)

fig.tight_layout()
out = Path("figures/FIG_CGM_detail.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved ->", out)
plt.close(fig)
