#!/usr/bin/env python3
"""
disagg_arena.py — 2×3 disaggregation arena figure.
Row 0: 1K in / 500 out token pair
Row 1: 20K in / 5K out token pair
Cols:  Throughput (tok/s) | ITL (ms) | E2E Latency (s)

Configs (color): 2P6D, 4P4D, 6P2D
Models  (shade): 9B=light, 35B=base, 80B=dark — 9 bars per concurrency group.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mc
import numpy as np
import pandas as pd
from pathlib import Path

mpl.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.serif': ['cmr10', 'Computer Modern Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.6,
    'patch.linewidth': 0.4,
    'figure.dpi': 150,
    'axes.formatter.use_mathtext': True,
})

DATA_DIR = Path.home() / 'git/inference_poc/paper/fig/csv'
OUT_DIR  = Path.home() / 'git/inference_poc/paper/fig'

TOKEN_PAIRS = ['1000_500', '20000_5000']
# Short annotation rendered inside each subplot (top-left corner)
INOUT_LABEL = {'1000_500': '1K / 500', '20000_5000': '20K / 5K'}

ARENA_COLORS = {'2P6D': '#1b7837', '4P4D': '#4393c3', '6P2D': '#d6604d'}
ARENA_LABELS = {'2P6D': '2P6D', '4P4D': '4P4D', '6P2D': '6P2D'}

MODELS       = ['9b', '35b', '80b']
MODEL_LABELS = {'9b': '9B', '35b': '35B', '80b': '80B'}
SHADE        = {'9b': 1.45, '35b': 1.0, '80b': 0.55}
SHADE_REF    = '#777777'   # neutral gray for model-shade legend swatches

METRICS    = ['tps', 'itl', 'e2e']
COL_TITLES = ['Throughput (tok/s)', 'Inter-Token Latency (ms)', 'End-to-End Latency (s)']

CONFIG_GAP = 0.07
TOTAL_SPAN = 0.82
BAR_WIDTH  = (TOTAL_SPAN - 2 * CONFIG_GAP) / 9

_half = (9 * BAR_WIDTH + 2 * CONFIG_GAP) / 2
OFFSETS = [
    -_half + (ci * CONFIG_GAP) + (ci * 3 + mi + 0.5) * BAR_WIDTH
    for ci in range(3) for mi in range(3)
]


def shade_color(hex_color, factor):
    rgb = np.array(mc.to_rgb(hex_color))
    if factor >= 1:
        return tuple(np.clip(rgb + (1 - rgb) * (factor - 1), 0, 1))
    return tuple(np.clip(rgb * factor, 0, 1))


def parse_col(col):
    p = col.split('_')
    model = p[1]
    if any(c in col for c in ('2P6D', '4P4D', '6P2D')):
        config, token_pair = p[5], f'{p[3]}_{p[4]}'
    else:
        config, token_pair = f'TP{p[3]}', f'{p[4]}_{p[5]}'
    return config, model, token_pair


def load(path):
    df = pd.read_csv(path)
    return df[[c for c in df.columns if c.strip() and not c.startswith('Unnamed')]]


def fix_ylim(ax):
    """Snap y upper limit to the first major tick strictly above the tallest bar."""
    ax.set_ylim(bottom=0)
    if not ax.patches:
        return
    data_max = max(p.get_height() for p in ax.patches)
    if data_max <= 0:
        return
    # Ticks currently visible (bounded by ylim); may not include the one above data_max
    ticks = sorted(t for t in ax.get_yticks() if t >= 0)
    above = [t for t in ticks if t > data_max]
    if above:
        ax.set_ylim(0, above[0])          # snap to nearest tick above data
    elif len(ticks) >= 2:
        step = ticks[-1] - ticks[-2]
        ax.set_ylim(0, ticks[-1] + step)  # extend by one interval
    else:
        ax.set_ylim(0, data_max * 1.25)


def tps_fmt(v, _):
    if v == 0:
        return '0'
    return f'{v/1000:g}K' if v >= 1000 else f'{v:.0f}'


def plot_row(axes, files, color_map, token_pair):
    configs = list(color_map.keys())
    for ax, metric in zip(axes, METRICS):
        df     = load(files[metric])
        x_cats = list(df['concurrency'].values)
        x_pos  = np.arange(len(x_cats))

        for ci, config in enumerate(configs):
            for mi, model in enumerate(MODELS):
                col = next(
                    (c for c in df.columns
                     if c != 'concurrency'
                     and parse_col(c) == (config, model, token_pair)),
                    None,
                )
                if col is None:
                    continue
                ax.bar(
                    x_pos + OFFSETS[ci * 3 + mi],
                    df[col].values,
                    width=BAR_WIDTH * 0.90,
                    color=shade_color(color_map[config], SHADE[model]),
                    edgecolor='white', linewidth=0.3, zorder=2,
                )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in x_cats])
        ax.set_xlim(-0.5, len(x_cats) - 0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#cccccc', linestyle=':', linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        if metric == 'tps':
            ax.yaxis.set_major_formatter(FuncFormatter(tps_fmt))
        fix_ylim(ax)

    # In/out label in the top-left corner of every subplot in the row
    for ax in axes:
        ax.text(0.03, 0.97, INOUT_LABEL[token_pair],
                transform=ax.transAxes,
                va='top', ha='left', fontsize=6.5,
                color='#555555', style='italic', zorder=5)


ARENA_FILES = {m: DATA_DIR / f'disagg_arena_{m}.csv' for m in METRICS}

fig, axes = plt.subplots(2, 3, figsize=(8.5, 3.0))

for row_idx, tp in enumerate(TOKEN_PAIRS):
    plot_row(axes[row_idx], ARENA_FILES, ARENA_COLORS, tp)

# Titles and xlabels on every row
for row_axes in axes:
    for ax, title in zip(row_axes, COL_TITLES):
        ax.set_title(title, pad=3)
    for ax in row_axes:
        ax.set_xlabel('Concurrent Requests', labelpad=2)

# ── Spacing ───────────────────────────────────────────────────────────────────
fig.subplots_adjust(
    left=0.065, right=0.995,
    top=0.93,   bottom=0.20,
    hspace=0.60, wspace=0.26,
)

# ── Single shared legend ──────────────────────────────────────────────────────
h, l = [], []
for cfg, color in ARENA_COLORS.items():
    h.append(mpatches.Patch(facecolor=shade_color(color, SHADE['35b']), edgecolor='none'))
    l.append(ARENA_LABELS[cfg])
h.append(mpatches.Patch(visible=False)); l.append('')
for model in MODELS:
    h.append(mpatches.Patch(facecolor=shade_color(SHADE_REF, SHADE[model]), edgecolor='none'))
    l.append(MODEL_LABELS[model])

fig.legend(h, l,
           ncol=7, frameon=False,
           handlelength=1.0, handletextpad=0.45, columnspacing=0.9,
           loc='lower center', bbox_to_anchor=(0.5, 0.01))

p = OUT_DIR / 'disagg_arena.pdf'
fig.savefig(p, bbox_inches='tight', pad_inches=0, dpi=300)
print(f'Saved: {p}')
