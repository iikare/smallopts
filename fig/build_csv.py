#!/usr/bin/env python3
"""Regenerate the figure CSVs in csv/ from csv/all_data.csv (TPS via Little's law, ITL/E2E as p50)."""

import csv
from pathlib import Path

CSV_DIR = Path(__file__).resolve().parent / 'csv'
FIGURES = ('disagg_arena', 'disagg_budget', 'kvcache')
P50 = {'itl': 'itt_p50', 'e2e': 'e2e_p50'}

rows = list(csv.reader(open(CSV_DIR / 'all_data.csv')))
col = {name: i for i, name in enumerate(rows[0])}
# Retests are appended as later rows for the same (tag, concurrency); the last one wins.
runs = {(r[0], int(r[2])): r for r in rows[1:] if r[0]}


def value(tag, conc, metric):
    run = runs[(tag, conc)]
    if metric in P50:
        return float(run[col[P50[metric]]])
    # Closed-loop steady state: avoids the start-up/drain bias of the 100-request runs.
    return conc * float(run[col['output_tokens_actual_mean']]) / float(run[col['e2e_mean']])


for fig in FIGURES:
    for metric in ('tps', 'itl', 'e2e'):
        path = CSV_DIR / f'{fig}_{metric}.csv'
        table = list(csv.reader(open(path)))
        header = table[0]
        for row in table[1:]:
            conc = int(row[0])
            for j, tag in enumerate(header[1:], 1):
                tag = tag.strip()
                if not tag or j >= len(row):
                    continue
                v = value(tag, conc, metric)
                if metric == 'tps' and fig == 'disagg_budget' and '_fp8_1_' in tag:
                    v *= 8  # 8xTP1 = eight independent single-GPU instances
                row[j] = f'{v:.6g}'
        with open(path, 'w', newline='') as f:
            csv.writer(f, lineterminator='\n').writerows(table)
        print(f'wrote {path.name}')
