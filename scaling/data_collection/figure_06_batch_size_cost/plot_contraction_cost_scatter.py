#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Figure 6 scatter plot of per-contraction GPU time for different batch sizes.
Paper configuration: 100q/600g.
"""

import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gmean

mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['STIXGeneral', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset':  'stix',
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'xtick.major.size':  6,
    'ytick.major.size':  6,
    'xtick.minor.size':  3,
    'ytick.minor.size':  3,
    'xtick.top':         True,
    'ytick.right':       True,
})

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
POLICIES = ['2p1', '2p2', '2p3']
EXCLUDED_CKTS = {2}

COLORS = {2: '#8B0000', 5: '#CC5500', 10: '#003F72', 15: '#7B8894',
           20: '#76B900', 24: '#0071C5', 28: '#D4500F'}

CONFIGS = [
    (100, 600),
]


def parse_file(fpath, exclude_final_batch=True):
    with open(fpath) as f:
        content = f.read()
    chunks = re.split(r'RUN \d+/\d+ \| circuit_id=(\d+):', content)

    qpb_match = re.search(r'qubits_per_batch=([\d,]+)', content)
    qpb = [int(x) for x in qpb_match.group(1).split(',')] if qpb_match else []
    final_batch_idx = len(qpb) - 1 if qpb else -1

    all_per_contraction = {}
    for i in range(1, len(chunks), 2):
        ckt = int(chunks[i])
        if ckt in EXCLUDED_CKTS:
            continue
        body = chunks[i + 1] if i + 1 < len(chunks) else ''
        lines = re.findall(
            r'Batch (\d+): cpu=([\d.]+)s\s+gpu=([\d.]+)s\s+contractions=(\d+)', body)
        for b, cpu, gpu, nc in lines:
            bn, gt, n = int(b), float(gpu), int(nc)
            if n == 0:
                continue
            if exclude_final_batch and bn == final_batch_idx:
                continue
            pc_ms = gt / n * 1000
            q = qpb[bn] if bn < len(qpb) else 0
            if q not in all_per_contraction:
                all_per_contraction[q] = []
            all_per_contraction[q].append(pc_ms)

    return all_per_contraction


def parse_final_batch_only(fpath):
    """Extract per-contraction times for the final batch only."""
    with open(fpath) as f:
        content = f.read()
    qpb_match = re.search(r'qubits_per_batch=([\d,]+)', content)
    qpb = [int(x) for x in qpb_match.group(1).split(',')] if qpb_match else []
    if not qpb:
        return []
    final_idx = len(qpb) - 1

    chunks = re.split(r'RUN \d+/\d+ \| circuit_id=(\d+):', content)
    vals = []
    for i in range(1, len(chunks), 2):
        ckt = int(chunks[i])
        if ckt in EXCLUDED_CKTS:
            continue
        body = chunks[i + 1] if i + 1 < len(chunks) else ''
        lines = re.findall(
            r'Batch (\d+): cpu=([\d.]+)s\s+gpu=([\d.]+)s\s+contractions=(\d+)', body)
        for b, cpu, gpu, nc in lines:
            bn, gt, n = int(b), float(gpu), int(nc)
            if bn == final_idx and n > 0:
                vals.append(gt / n * 1000)
    return vals


def collect(nqubits, ngates, bs_values):
    data = {q: [] for q in bs_values}

    for target_q in bs_values:
        bs_label = f'{target_q}bs'
        for pol in POLICIES:
            fname = f'{nqubits}q_{ngates}g_100hs_{bs_label}_{pol}_ptsbe.txt'
            fpath = os.path.join(DATA_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            per_q = parse_file(fpath, exclude_final_batch=True)
            if target_q in per_q:
                data[target_q].extend(per_q[target_q])

    return {q: np.array(v) for q, v in data.items()}


def plot_single(nqubits, ngates, bs_values, data):
    fig, ax = plt.subplots(figsize=(11, 7))
    rng = np.random.default_rng(42)

    active_bs = [q for q in bs_values if len(data.get(q, [])) > 0 and np.any(data[q] > 0)]
    all_keys = active_bs

    geo_means = {}
    for i, q in enumerate(all_keys):
        vals = data[q]
        vals = vals[vals > 0]
        if len(vals) == 0:
            continue

        col = COLORS.get(q, '#333333')
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(i + jitter, vals, color=col + '30',
                   s=15, edgecolors='none', zorder=2)

        geo = gmean(vals)
        geo_means[q] = (i, geo)
        ax.plot(i, geo, 'D', color=col, markersize=14,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        ax.annotate(f'{geo:.1f}ms', (i, geo),
                    textcoords='offset points', xytext=(28, 0),
                    fontsize=12, fontweight='bold', color=col,
                    va='center')

    nonfinal_keys = [q for q in active_bs if q in geo_means]
    if len(nonfinal_keys) > 1:
        xs = [geo_means[q][0] for q in nonfinal_keys]
        ys = [geo_means[q][1] for q in nonfinal_keys]
        ax.plot(xs, ys, '--', color='#888888', linewidth=1.0, alpha=0.5, zorder=1)

    ax.set_yscale('log')
    ax.set_xticks(range(len(all_keys)))
    ax.set_xticklabels([str(q) for q in all_keys], fontsize=12)
    ax.set_ylabel('Per-batch contraction + sampling time (ms)', fontsize=14)
    ax.set_title(f'Per-Batch Cost by Batch Size\n({nqubits}q / {ngates}g, non-proportional)',
                 fontsize=16, pad=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.grid(True, which='major', ls='--', alpha=0.4)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    footnote = (
        "Scatter = per-batch (contraction + sampling) time measurements across circuits\n"
        f"Diamond = geometric mean  |  {nqubits}q / {ngates}g  |  100 hypersamples  |  "
        "Non-final batches only (final batch excluded)"
    )
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.005, footnote, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)

    out = os.path.join(DATA_DIR, f'plot_contraction_cost_{nqubits}q_{ngates}g.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close()


def _blend_hex(hex_color, alpha, bg=(1, 1, 1)):
    """Pre-blend a hex color with alpha onto a white background -> solid RGB tuple."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (r * alpha + bg[0] * (1 - alpha),
            g * alpha + bg[1] * (1 - alpha),
            b * alpha + bg[2] * (1 - alpha))


def plot_single_v2(nqubits, ngates, bs_values, data):
    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(12, 8))
        rng = np.random.default_rng(42)

        active_bs = [q for q in bs_values if len(data.get(q, [])) > 0 and np.any(data[q] > 0)]
        all_keys = active_bs

        geo_means = {}
        for i, q in enumerate(all_keys):
            vals = data[q]
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue

            col = COLORS.get(q, '#333333')
            scatter_col = _blend_hex(col, 0.65)
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(i + jitter, vals, color=scatter_col,
                       s=15, edgecolors='none', zorder=2)

            geo = gmean(vals)
            geo_means[q] = (i, geo)
            ax.plot(i, geo, 'D', color=col, markersize=14,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=5)
            ax.annotate(f'{geo:.1f}ms', (i, geo),
                        textcoords='offset points', xytext=(0, 18),
                        fontsize=22, fontweight='bold', color=col,
                        ha='center', va='bottom')

        nonfinal_keys = [q for q in active_bs if q in geo_means]
        if len(nonfinal_keys) > 1:
            xs = [geo_means[q][0] for q in nonfinal_keys]
            ys = [geo_means[q][1] for q in nonfinal_keys]
            trend_col = _blend_hex('#888888', 0.5)
            ax.plot(xs, ys, '--', color=trend_col, linewidth=1.0, zorder=1)

        ax.set_yscale('log')
        ax.set_xticks(range(len(all_keys)))
        ax.set_xticklabels([str(q) for q in all_keys], fontsize=22)
        ax.set_xlabel(r'Batch Size ($b_j$)', fontsize=28, labelpad=14)
        ax.set_ylabel('Per-batch contraction + sampling time (ms)', fontsize=28)
        ax.set_title(f'Per-Batch Cost by Batch Size\n({nqubits}q / {ngates}g, non-proportional)',
                     fontsize=28, pad=14)
        ax.tick_params(axis='x', pad=6)
        ax.tick_params(axis='y', labelsize=18)
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        fig.tight_layout(pad=1.5)

        out_pdf = os.path.join(DATA_DIR, f'plot_contraction_cost_{nqubits}q_{ngates}g_v2.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(DATA_DIR, f'plot_contraction_cost_{nqubits}q_{ngates}g_v2.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_png}')

        plt.close()


def plot_single_v3(nqubits, ngates, bs_values, data):
    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(12, 8))
        rng = np.random.default_rng(42)

        active_bs = [q for q in bs_values if len(data.get(q, [])) > 0 and np.any(data[q] > 0)]
        all_keys = active_bs

        geo_means = {}
        for i, q in enumerate(all_keys):
            vals = data[q]
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue

            col = COLORS.get(q, '#333333')
            scatter_col = _blend_hex(col, 0.65)
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(i + jitter, vals, color=scatter_col,
                       s=15, edgecolors='none', zorder=2)

            geo = gmean(vals)
            geo_means[q] = (i, geo)
            ax.plot(i, geo, 'D', color=col, markersize=15,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=5)
            ax.annotate(f'{geo:.1f}ms', (i, geo),
                        textcoords='offset points', xytext=(0, 18),
                        fontsize=22, fontweight='bold', color=col,
                        ha='center', va='bottom')

        nonfinal_keys = [q for q in active_bs if q in geo_means]
        if len(nonfinal_keys) > 1:
            xs = [geo_means[q][0] for q in nonfinal_keys]
            ys = [geo_means[q][1] for q in nonfinal_keys]
            trend_col = _blend_hex('#888888', 0.5)
            ax.plot(xs, ys, '--', color=trend_col, linewidth=2.0, zorder=1)

        ax.set_yscale('log')
        ax.set_xticks(range(len(all_keys)))
        ax.set_xticklabels([str(q) for q in all_keys], fontsize=23)
        ax.set_xlabel(r'Batch Size ($b_j$)', fontsize=28, labelpad=14)
        ax.set_ylabel('Per-batch contraction + sampling time (ms)', fontsize=28)
        ax.tick_params(axis='x', pad=6)
        ax.tick_params(axis='y', labelsize=20)
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        fig.tight_layout(pad=1.5)

        out_pdf = os.path.join(DATA_DIR, f'plot_contraction_cost_{nqubits}q_{ngates}g_v3.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(DATA_DIR, f'plot_contraction_cost_{nqubits}q_{ngates}g_v3.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_png}')

        plt.close()


def _plot_single_v4(nqubits, ngates, bs_values, data):
    """V4: larger fonts across all elements."""
    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(13, 9))
        rng = np.random.default_rng(42)

        active_bs = [q for q in bs_values if len(data.get(q, [])) > 0 and np.any(data[q] > 0)]
        all_keys = active_bs

        geo_means = {}
        for i, q in enumerate(all_keys):
            vals = data[q]
            vals = vals[vals > 0]
            if len(vals) == 0:
                continue

            col = COLORS.get(q, '#333333')
            scatter_col = _blend_hex(col, 0.65)
            jitter = rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(i + jitter, vals, color=scatter_col,
                       s=15, edgecolors='none', zorder=2)

            geo = gmean(vals)
            geo_means[q] = (i, geo)
            ax.plot(i, geo, 'D', color=col, markersize=17,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=5)
            ax.annotate(f'{geo:.1f}ms', (i, geo),
                        textcoords='offset points', xytext=(0, 95),
                        fontsize=26, fontweight='bold', color=col,
                        ha='center', va='bottom')

        nonfinal_keys = [q for q in active_bs if q in geo_means]
        if len(nonfinal_keys) > 1:
            xs = [geo_means[q][0] for q in nonfinal_keys]
            ys = [geo_means[q][1] for q in nonfinal_keys]
            trend_col = _blend_hex('#888888', 0.5)
            ax.plot(xs, ys, '--', color=trend_col, linewidth=3.0, zorder=1)

        ax.set_yscale('log')
        ax.set_xticks(range(len(all_keys)))
        ax.set_xticklabels([str(q) for q in all_keys], fontsize=27)
        ax.set_xlabel(r'Batch Size ($b_j$)', fontsize=32, labelpad=14)
        ax.set_ylabel('Per-batch contraction + sampling time (ms)', fontsize=32)
        ax.tick_params(axis='x', pad=6)
        ax.tick_params(axis='y', labelsize=28)
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        fig.tight_layout(pad=1.5)

        out_pdf = os.path.join(DATA_DIR, f'plot_contraction_cost_{nqubits}q_{ngates}g_v4.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(DATA_DIR, f'plot_contraction_cost_{nqubits}q_{ngates}g_v4.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_png}')

        plt.close()


if __name__ == '__main__':
    for nq, ng in CONFIGS:
        bs_values = [2, 5, 10, 15, 20, 24, 28] if nq == 100 and ng == 600 else [10, 15, 20, 24, 28]
        data = collect(nq, ng, bs_values)
        print(f'\n=== {nq}q / {ng}g ===')
        active = []
        for q in bs_values:
            vals = data[q][data[q] > 0] if len(data[q]) > 0 else np.array([])
            if len(vals) > 0:
                print(f'  {q}q: n={len(vals)}, geo-mean={gmean(vals):.1f}ms')
                active.append(q)
        if len(active) >= 2:
            ratio = gmean(data[active[-1]][data[active[-1]] > 0]) / gmean(data[active[0]][data[active[0]] > 0])
            print(f'  {active[0]}q-to-{active[-1]}q ratio: {ratio:.0f}x')
        _plot_single_v4(nq, ng, bs_values, data)
