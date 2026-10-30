#!/usr/bin/env python3
"""
roofline.py — Annotated roofline model for the background section.
Explains the roofline model as applied to multi-GPU LLM inference serving.

H200 SXM FP8 concrete anchors:
  Peak compute : 3958 TFLOPS
  HBM bandwidth: 4.8 TB/s  →  ridge ≈ 825 FLOPs/byte

Saves: roofline.svg
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

mpl.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 8.5,
    'axes.labelsize': 8.5,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.6,
    'figure.dpi': 150,
    'axes.formatter.use_mathtext': True,
})

OUT_DIR = Path(__file__).parent

PEAK  = 3958.0   # TFLOPS  (H200 FP8 tensor-core)
BW    = 4.8      # TB/s    (H200 HBM3e)
RIDGE = PEAK / BW  # ≈ 825 FLOPs/byte

ai = np.logspace(0, 4, 3000)
roofline = np.minimum(ai * BW, PEAK)

fig, ax = plt.subplots(figsize=(5.5, 3.8))

# ── Shaded regions ────────────────────────────────────────────────────────────
ai_mb = ai[ai <= RIDGE]
ai_cb = ai[ai >= RIDGE]

ax.fill_between(ai_mb, ai_mb * BW, 0.6, alpha=0.08, color='#2166ac', zorder=1)
ax.fill_between(ai_cb, PEAK,       0.6, alpha=0.08, color='#b2182b', zorder=1)

# ── Roofline curve ────────────────────────────────────────────────────────────
ax.loglog(ai, roofline, color='#111111', linewidth=1.8, zorder=5,
          solid_capstyle='round', solid_joinstyle='round')

# ── Ridge vertical guide ──────────────────────────────────────────────────────
ax.axvline(RIDGE, color='#999999', linestyle='--', linewidth=0.8, zorder=2)

# ── Axes (set limits before computing display transforms) ─────────────────────
ax.set_xlabel('Arithmetic Intensity $I$ (FLOPs/byte)')
ax.set_ylabel('Attainable Performance (TFLOPS)')
ax.set_xlim(1, 8000)
ax.set_ylim(1, 12000)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(which='major', color='#e0e0e0', linestyle=':', linewidth=0.4, zorder=0)
ax.grid(which='minor', color='#eeeeee', linestyle=':', linewidth=0.3, zorder=0)

# Compute the true display angle of the memory-bound slope BEFORE adding text.
# In a log-log plot with a non-square aspect ratio the visual angle differs
# substantially from 45°; we measure it from the actual data→display transform.
fig.tight_layout(pad=0.7)
fig.canvas.draw()
p1 = ax.transData.transform((10,  10  * BW))
p2 = ax.transData.transform((200, 200 * BW))
slope_angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────

# 1. Memory-bound slope arrow + label (angle matches the actual visual slope)
ax.annotate('',
    xy=(60, 60 * BW), xytext=(10, 10 * BW),
    arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.0), zorder=6)
ax.text(28, 40, 'Slope $= B_s$\n(memory bandwidth)',
        fontsize=7, color='#2166ac', ha='left', va='bottom',
        rotation=slope_angle, rotation_mode='anchor', zorder=7)

# 2. Peak compute ceiling label
ax.annotate('',
    xy=(3000, PEAK), xytext=(1200, PEAK),
    arrowprops=dict(arrowstyle='->', color='#b2182b', lw=0.8), zorder=6)
ax.text(3200, PEAK * 1.08,
        r'$P_{\mathrm{peak}} = 3{,}958$ TFLOPS (H200 FP8)',
        fontsize=7, color='#b2182b', ha='left', va='bottom', zorder=7)

# 3. Ridge point annotation — placed in the compute-bound area to avoid crowding
#    the memory-bound region; arrow curves back left to the kink.
ax.annotate(r'Ridge point' '\n' r'$I = P_{\mathrm{peak}} / B_s \approx 825$ FLOPs/byte',
    xy=(RIDGE, PEAK * 0.97),
    xytext=(RIDGE * 1.9, PEAK * 0.45),
    fontsize=7, color='#444444', ha='left', va='top',
    arrowprops=dict(arrowstyle='->', color='#444444', lw=0.8,
                    connectionstyle='arc3,rad=0.25'), zorder=7)

# 4. "Evaluated range" green band + double-headed bracket (C=8–64, AI=16–128)
AI_LO, AI_HI = 16, 128
ax.axvspan(AI_LO, AI_HI, alpha=0.18, color='#4dac26', zorder=3)

perf_hi = AI_HI * BW
ax.annotate('',
    xy=(AI_HI, perf_hi * 1.5), xytext=(AI_LO, perf_hi * 1.5),
    arrowprops=dict(arrowstyle='<->', color='#4dac26', lw=0.9), zorder=7)
ax.text((AI_LO * AI_HI) ** 0.5, perf_hi * 1.85,
        r'Evaluated' '\n' r'$C = 8$-$64$' '\n' r'$I \approx 16$-$128$',
        fontsize=6.8, color='#4dac26', ha='center', va='bottom', zorder=8)

# 5. "Increasing concurrency →" arrow along x-axis bottom
ax.annotate('', xy=(400, 2.2), xytext=(20, 2.2),
    arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8), zorder=6)
ax.text(90, 1.6, 'Increasing batch size / concurrency $C$',
        fontsize=6.8, color='#555555', ha='center', va='top', zorder=7)

# 6. "Requires C ≳ 400" — text only, in lower compute-bound region
ax.text(RIDGE * 3.2, 18,
        'Compute-bound requires\n$C \\gtrsim 400$\n(impractical for serving)',
        fontsize=6.5, color='#b2182b', ha='center', va='bottom', zorder=7)

# 7. Region labels (large, italic, inside shaded area)
ax.text(5, 1500, 'Memory-\nbandwidth\nbound',
        fontsize=8, color='#2166ac', ha='left', va='top',
        style='italic', zorder=8)
ax.text(2800, 120, 'Compute\nbound',
        fontsize=8, color='#b2182b', ha='center', va='bottom',
        style='italic', zorder=8)

# 8. Formula box at top-left
ax.text(1.3, 8000,
        r'$P = \min\left(I \cdot B_s,\ P_{\mathrm{peak}}\right)$',
        fontsize=8.5, color='#222222', ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#cccccc', linewidth=0.6), zorder=9)

# 9. "Attainable performance" label on the slope (angle matches the visual slope)
ax.text(RIDGE * 0.55, RIDGE * 0.55 * BW * 1.25,
        'Attainable\nperformance',
        fontsize=6.8, color='#333333', ha='center', va='bottom',
        rotation=slope_angle, rotation_mode='anchor', zorder=7)

for fmt in ('svg', 'pdf'):
    out = OUT_DIR / f'roofline.{fmt}'
    fig.savefig(out, format=fmt, bbox_inches='tight')
    print(f'Saved: {out}')
