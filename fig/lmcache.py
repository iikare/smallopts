#!/usr/bin/env python3
"""
lmcache.py — LMCache vs baseline comparison (TP8, all token pairs).
Produces: lmcache.pdf

Layout: 4×3 grid
  Rows: 1K/500 | 5K/2.5K | 10K/5K | 20K/5K
  Cols: Throughput (tok/s) | ITL (ms) | E2E Latency (s)

Bars per concurrency bin (6 total):
  [9B, 35B, 80B] Baseline  |gap|  [9B, 35B, 80B] LMCache

Outlier bars are clipped to the subplot non-outlier max and marked
with hatching + a small annotation showing the true value.
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

TP = '8'   # single TP degree for this figure

TOKEN_PAIRS = ['1000_500', '5000_2500', '10000_5000', '20000_5000']
INOUT_LABEL = {
    '1000_500':   '1K / 500',
    '5000_2500':  '5K / 2.5K',
    '10000_5000': '10K / 5K',
    '20000_5000': '20K / 5K',
}

MODELS       = ['9b', '35b', '80b']
MODEL_LABELS = {'9b': '9B', '35b': '35B', '80b': '80B'}

# Vivid but balanced palette — one color per model
MODEL_COLORS = {'9b': '#1b7837', '35b': '#4393c3', '80b': '#d6604d'}
SHADE_BASE   = 1.00   # LMCache — full color
SHADE_LIGHT  = 1.65   # Baseline — washed out
SHADE_REF    = '#555555'

METRICS    = ['tps', 'itl', 'e2e']
COL_TITLES = ['Throughput (tok/s)', 'Inter-Token Latency (ms)', 'End-to-End Latency (s)']

# Outlier registry: metric → {(tp, model, token_pair, is_lmcache): set of concurrencies}.
# Only E2E needs clipping — TPS/ITL values are small bars, not chart-breaking.
OUTLIERS: dict = {
    'e2e': {
        ('8', '80b', '10000_5000', True): {64},        # 344s vs baseline ~80s
    }
}

# Bar geometry: 2 groups (Baseline, LMCache) × 3 models = 6 bars
GROUP_GAP  = 0.08
TOTAL_SPAN = 0.86
BAR_WIDTH  = (TOTAL_SPAN - GROUP_GAP) / 6

_half = (6 * BAR_WIDTH + GROUP_GAP) / 2
OFFSETS = [
    -_half + gi * GROUP_GAP + (gi * 3 + mi + 0.5) * BAR_WIDTH
    for gi in range(2) for mi in range(3)
]


def shade_color(hex_color, factor):
    rgb = np.array(mc.to_rgb(hex_color))
    if factor >= 1:
        return tuple(np.clip(rgb + (1 - rgb) * (factor - 1), 0, 1))
    return tuple(np.clip(rgb * factor, 0, 1))


def parse_col(col):
    """Returns (tp, model, token_pair, is_lmcache) for a kvcache CSV column."""
    p = col.split('_')
    # h200_{model}_fp8_{tp}_{in}_{out}[_lmcache]
    model      = p[1]
    tp         = p[3]
    token_pair = f'{p[4]}_{p[5]}'
    is_lmcache = len(p) > 6 and p[6] == 'lmcache'
    return tp, model, token_pair, is_lmcache


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
    ticks = sorted(t for t in ax.get_yticks() if t >= 0)
    above = [t for t in ticks if t > data_max]
    if above:
        ax.set_ylim(0, above[0])
    elif len(ticks) >= 2:
        step = ticks[-1] - ticks[-2]
        ax.set_ylim(0, ticks[-1] + step)
    else:
        ax.set_ylim(0, data_max * 1.25)


def tps_fmt(v, _):
    if v == 0:
        return '0'
    return f'{v/1000:g}K' if v >= 1000 else f'{v:.0f}'


def get_clip_max(df, token_pair, metric):
    """Max non-outlier value across all series for this (TP, token_pair, metric) subplot."""
    metric_outliers = OUTLIERS.get(metric, {})
    vals = []
    for col in df.columns:
        if col == 'concurrency':
            continue
        try:
            t, m, tok, lmc = parse_col(col)
        except (IndexError, ValueError):
            continue
        if t != TP or tok != token_pair:
            continue
        out_set = metric_outliers.get((t, m, tok, lmc), set())
        for conc, v in zip(df['concurrency'], df[col]):
            if conc not in out_set and pd.notna(v) and v > 0:
                vals.append(v)
    return max(vals) if vals else None


def plot_row(axes, files, token_pair):
    for ax, metric in zip(axes, METRICS):
        df     = load(files[metric])
        x_cats = list(df['concurrency'])
        x_pos  = np.arange(len(x_cats))

        clip_max = get_clip_max(df, token_pair, metric)
        metric_outliers = OUTLIERS.get(metric, {})

        for gi, is_lmc in enumerate([False, True]):
            for mi, model in enumerate(MODELS):
                col = next(
                    (c for c in df.columns
                     if c != 'concurrency'
                     and parse_col(c) == (TP, model, token_pair, is_lmc)),
                    None,
                )
                if col is None:
                    continue

                shade   = SHADE_BASE if is_lmc else SHADE_LIGHT
                color   = shade_color(MODEL_COLORS[model], shade)
                offset  = OFFSETS[gi * 3 + mi]
                out_set = metric_outliers.get((TP, model, token_pair, is_lmc), set())

                for xi, conc in enumerate(x_cats):
                    vals = df.loc[df['concurrency'] == conc, col]
                    if len(vals) == 0:
                        continue
                    v      = float(vals.iloc[0])
                    is_out = conc in out_set

                    if is_out and clip_max:
                        # Taller than every unclipped bar so the outlier still reads as the largest
                        h = clip_max * 1.15
                        # hatch.color separates hatch line color from edgecolor so
                        # the bar border stays white (same visual width as regular bars)
                        with mpl.rc_context({'hatch.color': '#666666',
                                             'hatch.linewidth': 0.5}):
                            ax.bar(x_pos[xi] + offset, h,
                                   width=BAR_WIDTH * 0.88,
                                   color=color, hatch='////',
                                   edgecolor='white', linewidth=0.3, zorder=2)
                        # Single arrow only — no value text to avoid title collision
                        ax.text(x_pos[xi] + offset, h * 1.02,
                                '$\\uparrow$',
                                ha='center', va='bottom', fontsize=5,
                                color='#444444', zorder=3)
                    else:
                        ax.bar(x_pos[xi] + offset, v,
                               width=BAR_WIDTH * 0.88,
                               color=color, edgecolor='white',
                               linewidth=0.3, zorder=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in x_cats])
        ax.set_xlim(-0.5, len(x_cats) - 0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#cccccc', linestyle=':', linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        if metric == 'tps':
            ax.yaxis.set_major_formatter(FuncFormatter(tps_fmt))
        fix_ylim(ax)

    for ax in axes:
        ax.text(0.03, 0.97, INOUT_LABEL[token_pair],
                transform=ax.transAxes,
                va='top', ha='left', fontsize=6.5,
                color='#555555', style='italic', zorder=5)


files = {m: DATA_DIR / f'kvcache_{m}.csv' for m in METRICS}

fig, axes = plt.subplots(4, 3, figsize=(8.5, 5.5))

for row_idx, token_pair in enumerate(TOKEN_PAIRS):
    plot_row(axes[row_idx], files, token_pair)

for row_axes in axes:
    for ax, title in zip(row_axes, COL_TITLES):
        ax.set_title(title, pad=3)
    for ax in row_axes:
        ax.set_xlabel('Concurrent Requests', labelpad=2)

fig.subplots_adjust(
    left=0.065, right=0.995,
    top=0.955,  bottom=0.11,
    hspace=0.72, wspace=0.26,
)

# Legend: model color swatches + spacer + Baseline/LMCache shade swatches
h, l = [], []
for model in MODELS:
    h.append(mpatches.Patch(
        facecolor=shade_color(MODEL_COLORS[model], SHADE_BASE), edgecolor='none'))
    l.append(MODEL_LABELS[model])
h.append(mpatches.Patch(visible=False)); l.append('')
h.append(mpatches.Patch(
    facecolor=shade_color(SHADE_REF, SHADE_LIGHT), edgecolor='none'))
l.append('Baseline')
h.append(mpatches.Patch(
    facecolor=shade_color(SHADE_REF, SHADE_BASE), edgecolor='none'))
l.append('LMCache')

fig.legend(h, l, ncol=6, frameon=False,
           handlelength=1.0, handletextpad=0.45, columnspacing=0.9,
           loc='lower center', bbox_to_anchor=(0.5, 0.01))

p = OUT_DIR / 'lmcache.pdf'
fig.savefig(p, bbox_inches='tight', pad_inches=0, dpi=300)
print(f'Saved: {p}')
