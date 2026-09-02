#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Presentation-quality FBS sweep plots for 200-qubit circuits (Figure 4, panel A).

Generates two plots:
  final_throughput_advantage.png  — PTSBE/cuda-Q advantage vs gates, legend = fbs
  final_ptsbe_throughput.png      — raw PTSBE throughput vs gates, legend = fbs

Does NOT alter original plots.
"""

import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Academic / publication style ──────────────────────────────────────────────
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_03_DIR = os.path.join(SCRIPT_DIR, '..', 'figure_03_data_collection_speedup')

GATES = [200, 400, 600, 800, 1000]
FBS_LIST = [24, 26, 28]
NUM_CIRCUITS = 10

COLORS = {24: '#4477AA', 26: '#228833', 28: '#EE6677'}
COLORS_LIGHT = {k: c + '55' for k, c in COLORS.items()}
MARKERS = {24: 'o', 26: 's', 28: '^'}

FOOTNOTE_ADV = (
    "Markers = geometric mean, error bars = \u00b11 geometric std deviation  |  "
    "10 random circuits per config  |  100 hypersamples\n"
    "200 qubits, non-final batch size = 10  |  "
    "cuda-Q baseline from Figure 3 (non-final batch size = 10, final batch size = 28)  |  "
    "y-axis log scale"
)

FOOTNOTE_TPT = (
    "Markers = geometric mean, error bars = \u00b11 geometric std deviation  |  "
    "10 random circuits per config  |  100 hypersamples\n"
    "200 qubits, non-final batch size = 10  |  y-axis log scale"
)


def _extract_ptsbe(filepath):
    vals = {}
    current_ckt = None
    with open(filepath) as f:
        for line in f:
            m = re.search(r'circuit_id=(\d+)', line)
            if m:
                current_ckt = int(m.group(1))
            if current_ckt is not None and 'PTSBE shots/s:' in line and '+/-' not in line:
                v = float(line.split(':')[-1].strip())
                vals[current_ckt] = v
    return [vals.get(i, None) for i in range(NUM_CIRCUITS)]


def _extract_cudaq(filepath):
    shots = {}
    sample_time = {}
    current_ckt = None
    with open(filepath) as f:
        for line in f:
            m = re.search(r'circuit_id=(\d+)', line)
            if m:
                current_ckt = int(m.group(1))
            if current_ckt is None:
                continue
            ms = re.search(r'Total number of distinct CUDA-Q shots collected:\s+(\d+)', line)
            if ms:
                shots[current_ckt] = int(ms.group(1))
            mt = re.search(r'Sample time:\s+([\d.]+)s', line)
            if mt:
                sample_time[current_ckt] = float(mt.group(1))
    result = []
    for i in range(NUM_CIRCUITS):
        n = shots.get(i)
        t = sample_time.get(i)
        if n is not None and t is not None and t > 0:
            result.append(n / t)
        else:
            result.append(None)
    return result


def _collect_data():
    ptsbe = {}
    for g in GATES:
        for fbs in FBS_LIST:
            fname = f'200q_{g}g_100hs_10nfbs_{fbs}fbs_ptsbe.txt'
            path = os.path.join(SCRIPT_DIR, fname)
            ptsbe[(g, fbs)] = _extract_ptsbe(path)

    cudaq = {}
    for g in GATES:
        fname = f'200q_{g}g_100hs_cudaq.txt'
        path = os.path.join(FIGURE_03_DIR, fname)
        cudaq[g] = _extract_cudaq(path)

    return ptsbe, cudaq


def _geo_stats(arr):
    log_arr = np.log(arr)
    geo_mean = np.exp(np.mean(log_arr))
    geo_std = np.exp(np.std(log_arr))
    return geo_mean, geo_mean - geo_mean / geo_std, geo_mean * geo_std - geo_mean


def _plot_throughput(ptsbe):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    offsets = {24: -8, 26: 0, 28: 8}

    for fbs in FBS_LIST:
        xs, means, lo, hi = [], [], [], []
        for g in GATES:
            arr = np.array([v for v in ptsbe[(g, fbs)] if v is not None and v > 0])
            if len(arr) == 0:
                continue
            geo, lo_err, hi_err = _geo_stats(arr)
            xs.append(g)
            means.append(geo)
            lo.append(lo_err)
            hi.append(hi_err)

        if not xs:
            continue
        x = np.array(xs, dtype=float) + offsets[fbs]
        ax.errorbar(x, means, yerr=[lo, hi],
                    fmt=MARKERS[fbs] + '-', color=COLORS[fbs],
                    capsize=4, capthick=1.0, markersize=8,
                    linewidth=1.6, elinewidth=1.0,
                    markeredgecolor=COLORS[fbs], markeredgewidth=0.8,
                    ecolor=COLORS_LIGHT[fbs],
                    label=f'Final batch size = {fbs}', zorder=3)

    ax.set_yscale('log')
    ax.set_xlabel('Number of Gates', fontsize=16)
    ax.set_ylabel('PTSBE Throughput (shots/s)', fontsize=16)
    ax.set_title('200-Qubit PTSBE Throughput vs Gates\nFinal-Batch-Size Sweep',
                 fontsize=17, pad=14)

    ax.set_xticks(GATES)
    ax.set_xticklabels([str(g) for g in GATES], fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which='major', ls='--', alpha=0.4)

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    leg = ax.legend(title='Final Batch Size', fontsize=12, title_fontsize=13,
                    framealpha=0.95, loc='best', fancybox=False,
                    edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    fig.text(0.5, 0.005, FOOTNOTE_TPT, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.subplots_adjust(bottom=0.14)

    out = os.path.join(SCRIPT_DIR, 'final_ptsbe_throughput.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _plot_advantage(ptsbe, cudaq):
    fig, ax = plt.subplots(figsize=(10, 7.5))
    offsets = {24: -8, 26: 0, 28: 8}

    for fbs in FBS_LIST:
        xs, means, lo, hi = [], [], [], []
        for g in GATES:
            ratios = []
            for i in range(NUM_CIRCUITS):
                p = ptsbe[(g, fbs)][i]
                c = cudaq[g][i]
                if p is not None and c is not None and p > 0 and c > 0:
                    ratios.append(p / c)
            if not ratios:
                continue
            arr = np.array(ratios)
            geo, lo_err, hi_err = _geo_stats(arr)
            xs.append(g)
            means.append(geo)
            lo.append(lo_err)
            hi.append(hi_err)

        if not xs:
            continue
        x = np.array(xs, dtype=float) + offsets[fbs]
        ax.errorbar(x, means, yerr=[lo, hi],
                    fmt=MARKERS[fbs] + '-', color=COLORS[fbs],
                    capsize=4, capthick=1.0, markersize=8,
                    linewidth=1.6, elinewidth=1.0,
                    markeredgecolor=COLORS[fbs], markeredgewidth=0.8,
                    ecolor=COLORS_LIGHT[fbs],
                    label=f'Final batch size = {fbs}', zorder=3)

    ax.set_yscale('log')
    ax.set_xlabel('Number of Gates', fontsize=16)
    ax.set_ylabel('Throughput Advantage  (PTSBE / cuda-Q)', fontsize=16)
    ax.set_title('200-Qubit Throughput Advantage vs Gates\nFinal-Batch-Size Sweep',
                 fontsize=17, pad=14)

    ax.set_xticks(GATES)
    ax.set_xticklabels([str(g) for g in GATES], fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which='major', ls='--', alpha=0.4)

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    for val, label in [(1e3, '1 K\u00d7'), (1e6, '1 M\u00d7')]:
        if ax.get_ylim()[0] <= val <= ax.get_ylim()[1] * 2:
            ax.axhline(y=val, color='#888888', ls='--', lw=0.8, alpha=0.6, zorder=1)
            ax.text(1.02, val, label, transform=ax.get_yaxis_transform(),
                    ha='left', va='center', fontsize=10, color='#333333',
                    fontweight='bold', clip_on=False,
                    bbox=dict(facecolor='white', edgecolor='#999999',
                              boxstyle='round,pad=0.25', linewidth=0.6, alpha=0.9))

    leg = ax.legend(title='Final Batch Size', fontsize=12, title_fontsize=13,
                    framealpha=0.95, loc='best', fancybox=False,
                    edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    fig.text(0.5, 0.005, FOOTNOTE_ADV, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.subplots_adjust(bottom=0.16)

    out = os.path.join(SCRIPT_DIR, 'final_throughput_advantage.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _ptsbe_success_rates():
    """Return {(g, fbs): success_fraction} based on PTSBE log parsing."""
    rates = {}
    for g in GATES:
        for fbs in FBS_LIST:
            fname = f'200q_{g}g_100hs_10nfbs_{fbs}fbs_ptsbe.txt'
            path = os.path.join(SCRIPT_DIR, fname)
            if not os.path.isfile(path):
                continue
            attempted = 0
            succeeded = 0
            in_run = False
            run_aborted = False
            run_has_timing = False
            with open(path) as f:
                for line in f:
                    if re.search(r'circuit_id=(\d+)', line):
                        if in_run:
                            attempted += 1
                            if not run_aborted and run_has_timing:
                                succeeded += 1
                        in_run = True
                        run_aborted = False
                        run_has_timing = False
                    if 'ABORT' in line:
                        run_aborted = True
                    if re.search(r'time_contraction_loop\s*:\s*[\d.]+s', line):
                        run_has_timing = True
                if in_run:
                    attempted += 1
                    if not run_aborted and run_has_timing:
                        succeeded += 1
            rates[(g, fbs)] = succeeded / max(attempted, 1)
    return rates


def _plot_throughput_paper(ptsbe):
    """Paper version: ratio plot of throughput relative to fbs=24 baseline, with hollow markers."""
    paper_colors = {26: '#7B8894', 28: '#76B900'}
    paper_colors_light = {k: c + '55' for k, c in paper_colors.items()}

    success = _ptsbe_success_rates()
    hollow = {(g, fbs) for (g, fbs), rate in success.items() if rate < 0.9}

    baseline_geo = {}
    for g in GATES:
        arr = np.array([v for v in ptsbe[(g, 24)] if v is not None and v > 0])
        if len(arr) > 0:
            baseline_geo[g] = np.exp(np.mean(np.log(arr)))

    fig, ax = plt.subplots(figsize=(10, 7.5))
    offsets = {26: -6, 28: 6}

    footnote = (
        "Markers = geometric mean of per-circuit throughput / geo-mean(fbs=24), "
        "error bars = \u00b11 geometric std deviation  |  "
        "10 random circuits per config\n"
        "200 qubits, non-final batch size = 10, 100 hypersamples  |  "
        "y-axis log scale  |  Baseline (fbs=24) shown as solid line at 1.0"
    )

    plotted_fbs = []
    for fbs in [26, 28]:
        xs, means, lo, hi, gs = [], [], [], [], []
        for g in GATES:
            if g not in baseline_geo or baseline_geo[g] == 0:
                continue
            if success.get((g, fbs), 0) == 0:
                continue
            vals = [v for v in ptsbe[(g, fbs)] if v is not None and v > 0]
            base_vals = [v for v in ptsbe[(g, 24)] if v is not None and v > 0]
            if not vals or not base_vals:
                continue
            ratios = [v / baseline_geo[g] for v in vals]
            arr = np.array(ratios)
            geo, lo_err, hi_err = _geo_stats(arr)
            xs.append(g)
            means.append(geo)
            lo.append(lo_err)
            hi.append(hi_err)
            gs.append(g)

        if not xs:
            continue
        plotted_fbs.append(fbs)
        x = np.array(xs, dtype=float) + offsets[fbs]
        ax.errorbar(x, means, yerr=[lo, hi],
                    fmt='-', color=paper_colors[fbs],
                    capsize=4, capthick=1.0,
                    linewidth=1.6, elinewidth=1.0,
                    ecolor=paper_colors_light[fbs],
                    zorder=3)

        for xi, yi, gi in zip(x, means, gs):
            is_hollow = (gi, fbs) in hollow
            ax.plot(xi, yi, marker=MARKERS[fbs], markersize=8,
                    color=paper_colors[fbs],
                    markeredgecolor=paper_colors[fbs], markeredgewidth=1.2,
                    markerfacecolor='white' if is_hollow else paper_colors[fbs],
                    zorder=4)

    ax.axhline(y=1.0, color='#003F72', ls='-', lw=1.5, alpha=0.7, zorder=1)
    ax.text(1.02, 1.0, 'fbs = 24\n(baseline)', transform=ax.get_yaxis_transform(),
            ha='left', va='center', fontsize=10, color='#003F72',
            fontweight='bold', clip_on=False,
            bbox=dict(facecolor='white', edgecolor='#003F72',
                      boxstyle='round,pad=0.25', linewidth=0.6, alpha=0.9))

    ax.set_yscale('log')
    ax.set_xlabel('Number of Gates', fontsize=16)
    ax.set_ylabel('Throughput Ratio  (fbs / fbs=24 baseline)', fontsize=16)
    ax.set_title('200-Qubit PTSBE Throughput Ratio vs Gates\nNormalized to Final Batch Size = 24',
                 fontsize=17, pad=14)

    ax.set_xticks(GATES)
    ax.set_xticklabels([str(g) for g in GATES], fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which='major', ls='--', alpha=0.4)

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    legend_handles = []
    for fbs in plotted_fbs:
        legend_handles.append(Line2D(
            [0], [0], color=paper_colors[fbs], marker=MARKERS[fbs],
            markersize=8, linewidth=1.6,
            markeredgecolor=paper_colors[fbs], markeredgewidth=1.2,
            markerfacecolor=paper_colors[fbs],
            label=f'Final batch size = {fbs}'))
    if hollow:
        legend_handles.append(Line2D(
            [0], [0], color='gray', marker='o', markersize=8,
            linewidth=0, markeredgecolor='gray', markeredgewidth=1.2,
            markerfacecolor='white', label='< 90% success'))
    leg = ax.legend(handles=legend_handles, title='Final Batch Size',
                    fontsize=12, title_fontsize=13,
                    framealpha=0.95, loc='best', fancybox=False,
                    edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    fig.text(0.5, 0.005, footnote, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.subplots_adjust(bottom=0.16)

    out = os.path.join(SCRIPT_DIR, 'paper_ptsbe_throughput_v1.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _plot_throughput_paper_v2(ptsbe):
    """Paper version v2: absolute throughput with NVIDIA palette, with hollow markers."""
    paper_colors = {24: '#003F72', 26: '#7B8894', 28: '#76B900'}
    paper_colors_light = {k: c + '55' for k, c in paper_colors.items()}

    success = _ptsbe_success_rates()
    hollow = {(g, fbs) for (g, fbs), rate in success.items() if rate < 0.9}

    fig, ax = plt.subplots(figsize=(10, 7.5))
    offsets = {24: -8, 26: 0, 28: 8}

    paper_footnote = (
        "Markers = geometric mean, error bars = \u00b11 geometric std deviation  |  "
        "10 random circuits per config  |  100 hypersamples\n"
        "200 qubits, non-final batch size = 10  |  y-axis log scale"
    )

    plotted_fbs_v2 = []
    for fbs in FBS_LIST:
        xs, means, lo, hi, gs = [], [], [], [], []
        for g in GATES:
            if success.get((g, fbs), 0) == 0:
                continue
            arr = np.array([v for v in ptsbe[(g, fbs)] if v is not None and v > 0])
            if len(arr) == 0:
                continue
            geo, lo_err, hi_err = _geo_stats(arr)
            xs.append(g)
            means.append(geo)
            lo.append(lo_err)
            hi.append(hi_err)
            gs.append(g)

        if not xs:
            continue
        plotted_fbs_v2.append(fbs)
        x = np.array(xs, dtype=float) + offsets[fbs]
        ax.errorbar(x, means, yerr=[lo, hi],
                    fmt='-', color=paper_colors[fbs],
                    capsize=4, capthick=1.0,
                    linewidth=1.6, elinewidth=1.0,
                    ecolor=paper_colors_light[fbs],
                    zorder=3)

        for xi, yi, gi in zip(x, means, gs):
            is_hollow = (gi, fbs) in hollow
            ax.plot(xi, yi, marker=MARKERS[fbs], markersize=8,
                    color=paper_colors[fbs],
                    markeredgecolor=paper_colors[fbs], markeredgewidth=1.2,
                    markerfacecolor='white' if is_hollow else paper_colors[fbs],
                    zorder=4)

    ax.set_yscale('log')
    ax.set_xlabel('Number of Gates', fontsize=16)
    ax.set_ylabel('PTSBE Throughput (shots/s)', fontsize=16)
    ax.set_title('200-Qubit PTSBE Throughput vs Gates\nFinal-Batch-Size Sweep',
                 fontsize=17, pad=14)

    ax.set_xticks(GATES)
    ax.set_xticklabels([str(g) for g in GATES], fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which='major', ls='--', alpha=0.4)

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    legend_handles_v2 = []
    for fbs in plotted_fbs_v2:
        legend_handles_v2.append(Line2D(
            [0], [0], color=paper_colors[fbs], marker=MARKERS[fbs],
            markersize=8, linewidth=1.6,
            markeredgecolor=paper_colors[fbs], markeredgewidth=1.2,
            markerfacecolor=paper_colors[fbs],
            label=f'Final batch size = {fbs}'))
    if hollow:
        legend_handles_v2.append(Line2D(
            [0], [0], color='gray', marker='o', markersize=8,
            linewidth=0, markeredgecolor='gray', markeredgewidth=1.2,
            markerfacecolor='white', label='< 90% success'))
    leg = ax.legend(handles=legend_handles_v2, title='Final Batch Size',
                    fontsize=12, title_fontsize=13,
                    framealpha=0.95, loc='best', fancybox=False,
                    edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    fig.text(0.5, 0.005, paper_footnote, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.subplots_adjust(bottom=0.16)

    out = os.path.join(SCRIPT_DIR, 'paper_ptsbe_throughput_v2.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _blend_hex(hex_color, alpha, bg=(1, 1, 1)):
    """Pre-blend a hex color with alpha onto a white background -> solid RGB tuple."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (r * alpha + bg[0] * (1 - alpha),
            g * alpha + bg[1] * (1 - alpha),
            b * alpha + bg[2] * (1 - alpha))


def _plot_throughput_paper_v2_v2(ptsbe, include_title=True):
    paper_colors = {24: '#003F72', 26: '#7B8894', 28: '#76B900'}

    success = _ptsbe_success_rates()
    hollow = {(g, fbs) for (g, fbs), rate in success.items() if rate < 0.9}

    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(12, 8))
        offsets = {24: -8, 26: 0, 28: 8}

        plotted_fbs = []
        for fbs in FBS_LIST:
            xs, means, lo, hi, gs = [], [], [], [], []
            for g in GATES:
                if success.get((g, fbs), 0) == 0:
                    continue
                arr = np.array([v for v in ptsbe[(g, fbs)] if v is not None and v > 0])
                if len(arr) == 0:
                    continue
                geo, lo_err, hi_err = _geo_stats(arr)
                xs.append(g)
                means.append(geo)
                lo.append(lo_err)
                hi.append(hi_err)
                gs.append(g)

            if not xs:
                continue
            plotted_fbs.append(fbs)
            x = np.array(xs, dtype=float) + offsets[fbs]
            ebar_col = _blend_hex(paper_colors[fbs], 0.45)
            ax.errorbar(x, means, yerr=[lo, hi],
                        fmt='-', color=paper_colors[fbs],
                        capsize=4, capthick=1.0,
                        linewidth=2.6, elinewidth=1.0,
                        ecolor=ebar_col, zorder=3)

            for xi, yi, gi in zip(x, means, gs):
                is_hollow = (gi, fbs) in hollow
                ax.plot(xi, yi, marker=MARKERS[fbs], markersize=11,
                        color=paper_colors[fbs],
                        markeredgecolor=paper_colors[fbs], markeredgewidth=1.2,
                        markerfacecolor='white' if is_hollow else paper_colors[fbs],
                        zorder=4)

        ax.set_yscale('log')
        ax.set_xlabel('Number of Gates', fontsize=28)
        ax.set_ylabel('PTSBE Throughput (shots/s)', fontsize=28)
        if include_title:
            ax.set_title('200-Qubit PTSBE Throughput vs Gates\nFinal-Batch-Size Sweep',
                         fontsize=28, pad=14)

        ax.set_xticks(GATES)
        ax.set_xticklabels([str(g) for g in GATES], fontsize=23)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        legend_handles = []
        for fbs in plotted_fbs:
            legend_handles.append(Line2D(
                [0], [0], color=paper_colors[fbs], marker=MARKERS[fbs],
                markersize=11, linewidth=2.6,
                markeredgecolor=paper_colors[fbs], markeredgewidth=1.2,
                markerfacecolor=paper_colors[fbs],
                label=f'Final batch size = {fbs}'))
        if hollow:
            legend_handles.append(Line2D(
                [0], [0], color='gray', marker='o', markersize=11,
                linewidth=0, markeredgecolor='gray', markeredgewidth=1.2,
                markerfacecolor='white', label='< 90% success'))
        leg = ax.legend(handles=legend_handles, title='Final Batch Size',
                        fontsize=23, title_fontsize=24,
                        framealpha=1.0, loc='best', fancybox=False,
                        edgecolor='black')
        leg.get_frame().set_linewidth(0.6)

        fig.tight_layout(pad=1.5)

        suffix = 'v2_v2' if include_title else 'v2_v3'
        out_pdf = os.path.join(SCRIPT_DIR, f'paper_ptsbe_throughput_{suffix}.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(SCRIPT_DIR, f'paper_ptsbe_throughput_{suffix}.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_png}')

        plt.close(fig)


def _plot_advantage_paper(ptsbe, cudaq, include_title=True):
    """Paper-quality throughput advantage plot (PTSBE / cuda-Q)."""
    paper_colors = {24: '#003F72', 26: '#7B8894', 28: '#76B900'}

    success = _ptsbe_success_rates()
    hollow = {(g, fbs) for (g, fbs), rate in success.items() if rate < 0.9}

    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(12, 8))
        offsets = {24: -8, 26: 0, 28: 8}

        plotted_fbs = []
        for fbs in FBS_LIST:
            xs, means, lo, hi, gs = [], [], [], [], []
            for g in GATES:
                if success.get((g, fbs), 0) == 0:
                    continue
                ratios = []
                for i in range(NUM_CIRCUITS):
                    p = ptsbe[(g, fbs)][i]
                    c = cudaq[g][i]
                    if p is not None and c is not None and p > 0 and c > 0:
                        ratios.append(p / c)
                if not ratios:
                    continue
                arr = np.array(ratios)
                geo, lo_err, hi_err = _geo_stats(arr)
                xs.append(g)
                means.append(geo)
                lo.append(lo_err)
                hi.append(hi_err)
                gs.append(g)

            if not xs:
                continue
            plotted_fbs.append(fbs)
            x = np.array(xs, dtype=float) + offsets[fbs]
            ebar_col = _blend_hex(paper_colors[fbs], 0.45)
            ax.errorbar(x, means, yerr=[lo, hi],
                        fmt='-', color=paper_colors[fbs],
                        capsize=4, capthick=1.0,
                        linewidth=1.6, elinewidth=1.0,
                        ecolor=ebar_col, zorder=3)

            for xi, yi, gi in zip(x, means, gs):
                is_hollow = (gi, fbs) in hollow
                ax.plot(xi, yi, marker=MARKERS[fbs], markersize=10,
                        color=paper_colors[fbs],
                        markeredgecolor=paper_colors[fbs], markeredgewidth=1.2,
                        markerfacecolor='white' if is_hollow else paper_colors[fbs],
                        zorder=4)

        ax.set_yscale('log')
        ax.set_xlabel(r'Number of Gates ($g$)', fontsize=28)
        ax.set_ylabel('Data Collection Speedup (PTSBE / cuda-Q)', fontsize=28)
        if include_title:
            ax.set_title('200-Qubit Data Collection Speedup vs Gates\nFinal-Batch-Size Sweep',
                         fontsize=28, pad=14)

        ax.set_xticks(GATES)
        ax.set_xticklabels([str(g) for g in GATES], fontsize=23)
        ax.tick_params(axis='y', labelsize=20)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        for val, label in [(1e3, '1 K\u00d7'), (1e6, '1 M\u00d7')]:
            if ax.get_ylim()[0] <= val <= ax.get_ylim()[1] * 2:
                ax.axhline(y=val, color='#888888', ls='--', lw=0.8, alpha=0.6, zorder=1)
                ax.text(1.02, val, label, transform=ax.get_yaxis_transform(),
                        ha='left', va='center', fontsize=14, color='#333333',
                        fontweight='bold', clip_on=False,
                        bbox=dict(facecolor='white', edgecolor='#999999',
                                  boxstyle='round,pad=0.25', linewidth=0.6, alpha=0.9))

        legend_handles = []
        for fbs in plotted_fbs:
            legend_handles.append(Line2D(
                [0], [0], color=paper_colors[fbs], marker=MARKERS[fbs],
                markersize=10, linewidth=1.6,
                markeredgecolor=paper_colors[fbs], markeredgewidth=1.2,
                markerfacecolor=paper_colors[fbs],
                label=r'$b_f$ = ' + str(fbs)))
        if hollow:
            legend_handles.append(Line2D(
                [0], [0], color='gray', marker='o', markersize=10,
                linewidth=0, markeredgecolor='gray', markeredgewidth=1.2,
                markerfacecolor='white', label='< 90% success'))
        leg = ax.legend(handles=legend_handles, title=r'Final Batch Size ($b_f$)',
                        fontsize=23, title_fontsize=24,
                        framealpha=1.0, loc='best', fancybox=False,
                        edgecolor='black')
        leg.get_frame().set_linewidth(0.6)

        fig.tight_layout(pad=1.5)

        suffix = 'speedup_v1' if include_title else 'speedup_v2'
        out_pdf = os.path.join(SCRIPT_DIR, f'paper_data_collection_{suffix}.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(SCRIPT_DIR, f'paper_data_collection_{suffix}.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_png}')

        plt.close(fig)


def main():
    ptsbe, _ = _collect_data()
    _plot_throughput_paper_v4(ptsbe)


def _plot_throughput_paper_v4(ptsbe):
    """V4: larger fonts across all elements."""
    paper_colors = {24: '#003F72', 26: '#7B8894', 28: '#76B900'}
    success = _ptsbe_success_rates()
    hollow = {(g, fbs) for (g, fbs), rate in success.items() if rate < 0.9}

    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(13, 9))
        offsets = {24: -8, 26: 0, 28: 8}

        plotted_fbs = []
        for fbs in FBS_LIST:
            xs, means, lo, hi, gs = [], [], [], [], []
            for g in GATES:
                if success.get((g, fbs), 0) == 0:
                    continue
                arr = np.array([v for v in ptsbe[(g, fbs)] if v is not None and v > 0])
                if len(arr) == 0:
                    continue
                geo, lo_err, hi_err = _geo_stats(arr)
                xs.append(g)
                means.append(geo)
                lo.append(lo_err)
                hi.append(hi_err)
                gs.append(g)

            if not xs:
                continue
            plotted_fbs.append(fbs)
            x = np.array(xs, dtype=float) + offsets[fbs]
            ebar_col = _blend_hex(paper_colors[fbs], 0.45)
            ax.errorbar(x, means, yerr=[lo, hi],
                        fmt='-', color=paper_colors[fbs],
                        capsize=5, capthick=2.0,
                        linewidth=3.6, elinewidth=3.6,
                        ecolor=ebar_col, zorder=3)

            for xi, yi, gi in zip(x, means, gs):
                is_hollow = (gi, fbs) in hollow
                ax.plot(xi, yi, marker=MARKERS[fbs], markersize=15,
                        color=paper_colors[fbs],
                        markeredgecolor=paper_colors[fbs], markeredgewidth=1.4,
                        markerfacecolor='white' if is_hollow else paper_colors[fbs],
                        zorder=4)

        ax.set_yscale('log')
        ax.set_xlabel(r'Number of Gates ($g$)', fontsize=32)
        ax.set_ylabel('PTSBE Throughput (shots/s)', fontsize=32)

        ax.set_xticks(GATES)
        ax.set_xticklabels([str(g) for g in GATES], fontsize=27)
        ax.tick_params(axis='y', labelsize=28)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        legend_handles = []
        for fbs in plotted_fbs:
            legend_handles.append(Line2D(
                [0], [0], color=paper_colors[fbs], marker=MARKERS[fbs],
                markersize=15, linewidth=3.6,
                markeredgecolor=paper_colors[fbs], markeredgewidth=1.4,
                markerfacecolor=paper_colors[fbs],
                label=r'$b_f$ = ' + str(fbs)))
        if hollow:
            legend_handles.append(Line2D(
                [0], [0], color='gray', marker='o', markersize=12,
                linewidth=0, markeredgecolor='gray', markeredgewidth=1.4,
                markerfacecolor='white', label='< 90% success'))
        leg = ax.legend(handles=legend_handles, title=r'Final Batch Size ($b_f$)',
                        fontsize=26, title_fontsize=27,
                        framealpha=1.0, loc='best', fancybox=False,
                        edgecolor='black')
        leg.get_frame().set_linewidth(0.6)

        fig.tight_layout(pad=1.5)

        for suffix, fmt in [('pdf', 'pdf'), ('png', 'png')]:
            out = os.path.join(SCRIPT_DIR, f'paper_ptsbe_throughput_v4.{suffix}')
            fig.savefig(out, format=fmt, dpi=600 if fmt == 'png' else None,
                        bbox_inches='tight', pad_inches=0.3)
            print(f'Saved  {out}')
        plt.close(fig)


def _plot_advantage_paper_v4(ptsbe, cudaq):
    """V4: larger fonts across all elements."""
    paper_colors = {24: '#003F72', 26: '#7B8894', 28: '#76B900'}
    success = _ptsbe_success_rates()
    hollow = {(g, fbs) for (g, fbs), rate in success.items() if rate < 0.9}

    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(13, 9))
        offsets = {24: -8, 26: 0, 28: 8}

        plotted_fbs = []
        for fbs in FBS_LIST:
            xs, means, lo, hi, gs = [], [], [], [], []
            for g in GATES:
                if success.get((g, fbs), 0) == 0:
                    continue
                ratios = []
                for i in range(NUM_CIRCUITS):
                    p = ptsbe[(g, fbs)][i]
                    c = cudaq[g][i]
                    if p is not None and c is not None and p > 0 and c > 0:
                        ratios.append(p / c)
                if not ratios:
                    continue
                arr = np.array(ratios)
                geo, lo_err, hi_err = _geo_stats(arr)
                xs.append(g)
                means.append(geo)
                lo.append(lo_err)
                hi.append(hi_err)
                gs.append(g)

            if not xs:
                continue
            plotted_fbs.append(fbs)
            x = np.array(xs, dtype=float) + offsets[fbs]
            ebar_col = _blend_hex(paper_colors[fbs], 0.45)
            ax.errorbar(x, means, yerr=[lo, hi],
                        fmt='-', color=paper_colors[fbs],
                        capsize=5, capthick=2.0,
                        linewidth=3.6, elinewidth=3.6,
                        ecolor=ebar_col, zorder=3)

            for xi, yi, gi in zip(x, means, gs):
                is_hollow = (gi, fbs) in hollow
                ax.plot(xi, yi, marker=MARKERS[fbs], markersize=15,
                        color=paper_colors[fbs],
                        markeredgecolor=paper_colors[fbs], markeredgewidth=1.4,
                        markerfacecolor='white' if is_hollow else paper_colors[fbs],
                        zorder=4)

        ax.set_yscale('log')
        ax.set_xlabel(r'Number of Gates ($g$)', fontsize=32)
        ax.set_ylabel('Data Collection Speedup (PTSBE / cuda-Q)', fontsize=32)

        ax.set_xticks(GATES)
        ax.set_xticklabels([str(g) for g in GATES], fontsize=27)
        ax.tick_params(axis='y', labelsize=28)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        legend_handles = []
        for fbs in plotted_fbs:
            legend_handles.append(Line2D(
                [0], [0], color=paper_colors[fbs], marker=MARKERS[fbs],
                markersize=15, linewidth=3.6,
                markeredgecolor=paper_colors[fbs], markeredgewidth=1.4,
                markerfacecolor=paper_colors[fbs],
                label=r'$b_f$ = ' + str(fbs)))
        if hollow:
            legend_handles.append(Line2D(
                [0], [0], color='gray', marker='o', markersize=12,
                linewidth=0, markeredgecolor='gray', markeredgewidth=1.4,
                markerfacecolor='white', label='< 90% success'))
        leg = ax.legend(handles=legend_handles, title=r'Final Batch Size ($b_f$)',
                        fontsize=26, title_fontsize=27,
                        framealpha=1.0, loc='best', fancybox=False,
                        edgecolor='black')
        leg.get_frame().set_linewidth(0.6)

        fig.tight_layout(pad=1.5)

        for suffix, fmt in [('pdf', 'pdf'), ('png', 'png')]:
            out = os.path.join(SCRIPT_DIR, f'paper_data_collection_speedup_v4.{suffix}')
            fig.savefig(out, format=fmt, dpi=600 if fmt == 'png' else None,
                        bbox_inches='tight', pad_inches=0.3)
            print(f'Saved  {out}')
        plt.close(fig)


if __name__ == '__main__':
    main()
