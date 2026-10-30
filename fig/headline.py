#!/usr/bin/env python3
"""
headline.py — single-panel headline figure, square aspect ratio.

Two visual sections in one axes:
  LEFT  — Disaggregation (8xTP1 and 2P6D, 1K/500 tokens), blue palette
  RIGHT — KV Cache Offload (LMCache 1K/500 and 20K/5K), red/salmon palette

Each section carries a shaded band for "prior work claimed" vs our observed bars.
All values normalised to TP8 = 1.0.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

mpl.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 8.5,
    'axes.labelsize': 8,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'patch.linewidth': 0.4,
    'figure.dpi': 150,
    'axes.formatter.use_mathtext': True,
})

DATA = Path.home() / 'git/inference_poc/paper/fig/csv'
OUT  = Path.home() / 'git/inference_poc/paper/fig'
CONC = 64

def load(name):
    df = pd.read_csv(DATA / name)
    return df[[c for c in df.columns if c.strip() and not c.startswith('Unnamed')]]

def at(df, c=CONC):
    return df[df['concurrency'] == c].squeeze()

budget = at(load('disagg_budget_tps.csv'))
kv     = at(load('kvcache_tps.csv'))

MODELS  = ['9b', '35b', '80b']
MLABELS = ['9B\n(dense)', '35B\n(MoE)', '80B\n(MoE)']

r_8tp1, r_2p6d, r_lmc1k, r_lmc20k = [], [], [], []
for m in MODELS:
    tp8_d = float(budget[f'h200_{m}_fp8_8_1000_500'])
    r_8tp1.append(float(budget[f'h200_{m}_fp8_1_1000_500'])        / tp8_d)
    r_2p6d.append(float(budget[f'h200_{m}_fp8_1000_500_2P6D_tp2']) / tp8_d)

    b1k  = float(kv[f'h200_{m}_fp8_8_1000_500'])
    l1k  = float(kv[f'h200_{m}_fp8_8_1000_500_lmcache'])
    b20k = float(kv[f'h200_{m}_fp8_8_20000_5000'])
    l20k = float(kv[f'h200_{m}_fp8_8_20000_5000_lmcache'])
    r_lmc1k.append(l1k  / b1k)
    r_lmc20k.append(l20k / b20k)

_all_vals = r_8tp1 + r_2p6d + r_lmc1k + r_lmc20k

# ── Layout ────────────────────────────────────────────────────────────────────
n   = len(MODELS)
X_D = np.array([0.6, 1.6, 2.6])   # disagg group centres
X_L = np.array([4.4, 5.4, 6.4])   # lmcache group centres
SEP = 3.5                           # x position of visual separator
bw  = 0.33

# Offset pairs for 2 bars per group
L_OFF, R_OFF = -0.20, +0.20

# Colours — RdBu palette matching prefill_decode diagram (P=blue, D=red)
C_8TP1  = '#08306b'   # dark navy
C_2P6D  = '#2166ac'   # medium blue
C_LMC1K = '#ef8a62'   # salmon
C_LMC20 = '#b2182b'   # dark red

BG_D = '#d9e8f5'   # light blue section fill
BG_L = '#fde4dc'   # light red section fill

YMAX = np.ceil(max(_all_vals) / 0.5) * 0.5 + 0.5   # tight ceiling above data
XMIN, XMAX = -0.4, 7.4

fig, ax = plt.subplots(figsize=(5.5, 5.5))

# ── Section background fills ──────────────────────────────────────────────────
ax.axvspan(XMIN, SEP - 0.05, color=BG_D, alpha=1.0, zorder=0)
ax.axvspan(SEP + 0.05, XMAX, color=BG_L, alpha=1.0, zorder=0)

# ── TP8 baseline ──────────────────────────────────────────────────────────────
ax.axhline(1.0, color='#333', linestyle='--', linewidth=1.0, zorder=2)
ax.text(XMAX - 0.05, 1.05, 'TP8 = 1.0', fontsize=6.5, color='#333',
        ha='right', va='bottom', zorder=6)

# ── Disaggregation bars ───────────────────────────────────────────────────────
b8 = ax.bar(X_D + L_OFF, r_8tp1, bw, color=C_8TP1, label=r'$8\times$TP1', zorder=3)
b2 = ax.bar(X_D + R_OFF, r_2p6d, bw, color=C_2P6D, label='2P6D disagg.', zorder=3)

# ── LMCache bars ──────────────────────────────────────────────────────────────
bk = ax.bar(X_L + L_OFF, r_lmc1k,  bw, color=C_LMC1K, label='LMCache (1K/500)',  zorder=3)
b4 = ax.bar(X_L + R_OFF, r_lmc20k, bw, color=C_LMC20,  label='LMCache (20K/5K)', zorder=3)

# ── "Prior work claimed" bands, section-local ─────────────────────────────────
# Disagg: prior work claimed 2-5x speedup from disaggregation
CLAIM_D_LO, CLAIM_D_HI = 2.0, 5.0
ax.fill_betweenx([CLAIM_D_LO, CLAIM_D_HI], XMIN, SEP - 0.05,
                  color='#2166ac', alpha=0.13, zorder=1)
ax.text(XMIN + 0.06, (CLAIM_D_LO + CLAIM_D_HI) / 2,
        'Prior work\nclaimed\n2--5x',
        fontsize=6.2, color='#08306b', ha='left', va='center',
        style='italic', zorder=5)

# LMCache: prior work claimed 2-14x improvement (dense models)
CLAIM_L_LO = 2.0
ax.fill_betweenx([CLAIM_L_LO, YMAX - 0.1], SEP + 0.05, XMAX,
                  color='#b2182b', alpha=0.10, zorder=1)
ax.text(XMAX - 0.08, CLAIM_L_LO + (YMAX - CLAIM_L_LO) / 2,
        'Prior work\nclaimed\nup to 15x',
        fontsize=6.2, color='#67001f', ha='right', va='center',
        style='italic', zorder=5)

# ── Section headers ───────────────────────────────────────────────────────────
ax.text((XMIN + SEP) / 2, YMAX * 0.995,
        'Disaggregation',
        fontsize=9.5, color='#08306b', ha='center', va='top',
        style='italic', zorder=6)
ax.text((SEP + XMAX) / 2, YMAX * 0.995,
        'KV Cache Offload',
        fontsize=9.5, color='#b2182b', ha='center', va='top',
        style='italic', zorder=6)

# ── Vertical separator ────────────────────────────────────────────────────────
ax.axvline(SEP, color='#888', linestyle=':', linewidth=0.9, zorder=4)

# ── Value labels ──────────────────────────────────────────────────────────────
def bar_label(bar_container, vals, fmt_fn, color, ymin_text=0.14):
    """Place label just above each bar; bump up if bar is very short."""
    for bar, v in zip(bar_container, vals):
        xc  = bar.get_x() + bar.get_width() / 2
        yv  = max(bar.get_height(), 0.0)
        yp  = max(yv, ymin_text) + 0.07
        ax.text(xc, yp, fmt_fn(v), ha='center', va='bottom',
                fontsize=6.5, color=color, zorder=7)

bar_label(b8, r_8tp1,   lambda v: f'{v:.1f}x',          C_8TP1)
bar_label(b2, r_2p6d,   lambda v: f'{v:.1f}x',          '#1a4069')
bar_label(bk, r_lmc1k,  lambda v: f'{(v-1)*100:+.0f}%', '#7a3800')
bar_label(b4, r_lmc20k, lambda v: f'{(v-1)*100:+.0f}%', '#67001f')

# ── Axes & formatting ─────────────────────────────────────────────────────────
ax.set_xticks(np.concatenate([X_D, X_L]))
ax.set_xticklabels(MLABELS + MLABELS)
ax.set_ylabel('Throughput relative to TP8')
ax.set_title(r'Performance at fixed 8-GPU budget, $C = 64$', pad=6)
ax.set_ylim(0, YMAX)
ax.set_xlim(XMIN, XMAX)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', color='#c0c0c0', linestyle=':', linewidth=0.4, zorder=0)
ax.set_axisbelow(True)

# ── Legend below the chart ───────────────────────────────────────────────────
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
          ncol=4, frameon=False,
          handlelength=1.0, handletextpad=0.35, columnspacing=0.9)

# ── Note: token lengths for disagg section ────────────────────────────────────
ax.text((XMIN + SEP) / 2, -0.42,
        '(1K/500 tok.)',
        fontsize=6.0, color='#444', ha='center', va='top',
        transform=ax.get_xaxis_transform(), zorder=6)

fig.tight_layout(pad=0.6)
for fmt in ('pdf', 'svg'):
    fig.savefig(OUT / f'headline.{fmt}', bbox_inches='tight')
    print(f'Saved: headline.{fmt}')
