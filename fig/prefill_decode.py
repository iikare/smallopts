#!/usr/bin/env python3
"""
prefill_decode.py — Disaggregated prefill/decode architecture overview.
Saves: prefill_decode.svg
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

mpl.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 8.5,
    'axes.linewidth': 0.6,
    'figure.dpi': 150,
    'axes.formatter.use_mathtext': True,
})

OUT_DIR = Path(__file__).parent

fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0, 5.8)
ax.axis('off')

# ── Helpers ───────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.0, rad=0.12, zo=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f'round,pad={rad}', facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=zo))

def arr(src, dst, c='#333', lw=1.2, ms=12, rad=0.0, zo=5):
    ax.annotate('', xy=dst, xytext=src,
        arrowprops=dict(arrowstyle='->', color=c, lw=lw,
            mutation_scale=ms, connectionstyle=f'arc3,rad={rad}'), zorder=zo)

def txt(x, y, s, **kw):
    ax.text(x, y, s, ha=kw.pop('ha', 'center'), va=kw.pop('va', 'center'), zorder=kw.pop('zo', 5), **kw)

# Palette
P_BG, P_ED = '#dce8f5', '#2166ac'
D_BG, D_ED = '#fde8e0', '#b2182b'
S_BG       = '#f5f2ec'
KV_C       = '#35783a'
G_P        = '#8fb8d8'
G_D        = '#f0a090'

# ── Scheduler ─────────────────────────────────────────────────────────────────
rbox(3.2, 4.55, 3.6, 0.85, S_BG, '#888', lw=0.9)
txt(5.0, 4.975, 'Scheduler / Router', fontsize=8, fontweight='bold', color='#444')

arr((-0.2, 4.975), (3.2, 4.975), c='#444', lw=1.0, ms=10)
txt(-0.25, 4.975, 'Requests', ha='right', fontsize=7, color='#444')

arr((3.6, 4.55), (2.25, 4.30), c='#666', lw=0.8, ms=9, rad=-0.15)
arr((6.4, 4.55), (7.85, 4.30), c='#666', lw=0.8, ms=9, rad=0.15)
txt(2.85, 4.57, 'route', fontsize=5.5, color='#666')
txt(7.15, 4.57, 'route', fontsize=5.5, color='#666')

# ── Prefill pool ──────────────────────────────────────────────────────────────
rbox(0.20, 0.30, 4.10, 3.80, P_BG, P_ED, lw=1.2)
txt(2.25, 3.97, 'Prefill (P) Pool', fontsize=8, fontweight='bold', color=P_ED)

# 2 nodes, centered in pool
for i, gx in enumerate([0.70, 2.35]):
    gy = 2.10
    rbox(gx, gy, 1.45, 1.65, '#c2d9ed', P_ED, lw=0.75, rad=0.08, zo=3)
    txt(gx+0.725, gy+1.49, f'Node P{i}', fontsize=6.0, fontweight='bold', color='#1a4d80')
    txt(gx+0.725, gy+1.26, r'$8\times$ H200 (TP=8)', fontsize=5.2, color='#1a4d80', style='italic')
    # 4×2 GPU chip grid
    for r in range(2):
        for c in range(4):
            rbox(gx+0.13+c*0.31, gy+0.14+r*0.52, 0.26, 0.40, G_P, P_ED, lw=0.3, rad=0.03, zo=4)

txt(2.25, 1.88, 'Compute-bound', fontsize=6.5, color=P_ED, style='italic')
txt(2.25, 1.58, 'Full prompt in one forward pass', fontsize=6.0, color='#1a4d80')
txt(2.25, 1.28, r'TTFT-critical $\cdot$ $I \gg$ ridge point', fontsize=5.8, color='#1a4d80')
txt(2.25, 0.65, '(P in disaggregated P/D config)', fontsize=5.5, color='#4a7096', style='italic')

# ── Decode pool ───────────────────────────────────────────────────────────────
rbox(5.80, 0.30, 4.10, 3.80, D_BG, D_ED, lw=1.2)
txt(7.85, 3.97, 'Decode (D) Pool', fontsize=8, fontweight='bold', color=D_ED)

# 6 nodes: 3 cols × 2 rows, centered
for row in range(2):
    for col in range(3):
        idx  = row * 3 + col
        gx   = 6.00 + col * 1.29
        gy   = 2.65 if row == 0 else 1.45
        rbox(gx, gy, 1.12, 1.10, '#f5c4b4', D_ED, lw=0.6, rad=0.07, zo=3)
        txt(gx+0.56, gy+0.94, f'Node D{idx}', fontsize=5.3, fontweight='bold', color='#7a1a1a')
        txt(gx+0.56, gy+0.73, r'$8\times$ H200', fontsize=4.8, color='#7a1a1a', style='italic')
        # 2×2 GPU chip grid
        for r2 in range(2):
            for c2 in range(2):
                rbox(gx+0.17+c2*0.44, gy+0.12+r2*0.28, 0.34, 0.20, G_D, D_ED, lw=0.3, rad=0.02, zo=4)

txt(7.85, 1.18, 'Memory-bandwidth bound', fontsize=6.5, color=D_ED, style='italic')
txt(7.85, 0.90, r'1 token/step (autoregressive)', fontsize=6.0, color='#7a1a1a')
txt(7.85, 0.63, r'ITL-critical $\cdot$ $I \approx 2C$ FLOPs/byte', fontsize=5.8, color='#7a1a1a')
txt(7.85, 0.48, '(D in disaggregated P/D config)', fontsize=5.5, color='#a04060', style='italic')

# ── KV cache transfer arrow ───────────────────────────────────────────────────
kv_y = 3.02
arr((4.30, kv_y), (5.80, kv_y), c=KV_C, lw=2.5, ms=16, zo=6)
txt(5.05, kv_y+0.28, 'KV Cache', fontsize=7, color=KV_C, fontweight='bold')
txt(5.05, kv_y-0.22, 'Transfer', fontsize=6.5, color=KV_C)
txt(5.05, kv_y-0.54, '(PCIe 5.0)', fontsize=5.5, color=KV_C, style='italic')

# ── Output tokens ─────────────────────────────────────────────────────────────
arr((9.90, kv_y), (10.38, kv_y), c='#444', lw=1.1, ms=10, zo=5)
txt(10.43, kv_y, 'Tokens', ha='left', fontsize=7, color='#444')

fig.tight_layout(pad=0.3)
out = OUT_DIR / 'prefill_decode.svg'
fig.savefig(out, format='svg', bbox_inches='tight')
print(f'Saved: {out}')
