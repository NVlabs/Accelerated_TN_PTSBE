#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Presentation-quality throughput-advantage plots (Figure 3 data).

Outputs:
  final_throughput_advantage_stddev_plot.png      — all qubits (x = qubits, legend = gates)
  final_throughput_advantage_200q_vs_gates.png    — 200q slice
  final_throughput_advantage_gates_axis.png       — all gates  (x = gates, legend = qubits)
"""

import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

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

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

QUBITS = [50, 75, 100, 150, 200]
GATES = [200, 400, 600, 800, 1000, 1200]

# Paul Tol's bright qualitative palette (colorblind-safe) — keyed by gate count
COLORS = {200: '#4477AA', 400: '#228833', 600: '#CCBB44',
          800: '#EE6677', 1000: '#AA3377', 1200: '#332288'}
COLORS_LIGHT = {g: c + '55' for g, c in COLORS.items()}

# Same palette re-keyed by qubit count for the gates-axis plot
COLORS_Q = {50: '#4477AA', 75: '#228833', 100: '#CCBB44',
            150: '#EE6677', 200: '#AA3377'}
COLORS_Q_LIGHT = {q: c + '55' for q, c in COLORS_Q.items()}

FOOTNOTE = (
    "Markers = geometric mean, error bars = \u00b11 geometric std deviation  |  "
    "10 random circuits per config  |  100 hypersamples\n"
    "Non-final batch size = 10 qubits, final batch size = 28 qubits  |  "
    "y-axis log scale"
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
                if v > 0:
                    vals[current_ckt] = v
    return vals


def _extract_cudaq(filepath):
    """Precise cuda-Q shots/s via distinct_shots / sample_time."""
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
    result = {}
    for ckt in set(shots) & set(sample_time):
        t = sample_time[ckt]
        if t > 0:
            result[ckt] = shots[ckt] / t
    return result


def _collect_advantages():
    advantages = defaultdict(list)
    for q in QUBITS:
        for g in GATES:
            pf = os.path.join(DATA_DIR, f'{q}q_{g}g_100hs_10nfbs_28fbs_ptsbe.txt')
            cf = os.path.join(DATA_DIR, f'{q}q_{g}g_100hs_cudaq.txt')
            if not os.path.isfile(pf) or not os.path.isfile(cf):
                continue
            ptsbe = _extract_ptsbe(pf)
            cudaq = _extract_cudaq(cf)
            for ckt in sorted(set(ptsbe) & set(cudaq)):
                p, c = ptsbe[ckt], cudaq[ckt]
                if p > 0 and c > 0:
                    advantages[(q, g)].append(p / c)
    return advantages


def _geo_stats(arr):
    log_arr = np.log(arr)
    geo_mean = np.exp(np.mean(log_arr))
    geo_std = np.exp(np.std(log_arr))
    return geo_mean, geo_mean - geo_mean / geo_std, geo_mean * geo_std - geo_mean


def _plot_all_qubits(adv):
    fig, ax = plt.subplots(figsize=(11, 7.5))

    markers = {200: 'o', 400: 's', 600: '^', 800: 'D', 1000: 'v', 1200: 'P'}
    offsets = {200: -3.75, 400: -2.25, 600: -0.75, 800: 0.75, 1000: 2.25, 1200: 3.75}

    for g in GATES:
        xs, means, lo, hi = [], [], [], []
        for q in QUBITS:
            ratios = adv.get((q, g))
            if not ratios:
                continue
            arr = np.array(ratios)
            geo, lo_e, hi_e = _geo_stats(arr)
            xs.append(q)
            means.append(geo)
            lo.append(lo_e)
            hi.append(hi_e)

        if not xs:
            continue

        x = np.array(xs, dtype=float) + offsets[g]
        ax.errorbar(x, means, yerr=[lo, hi],
                    fmt=markers[g] + '-', color=COLORS[g],
                    capsize=4, capthick=1.0, markersize=8,
                    linewidth=1.6, elinewidth=1.0,
                    markeredgecolor=COLORS[g], markeredgewidth=0.8,
                    ecolor=COLORS_LIGHT[g],
                    label=f'{g} gates', zorder=3)

    ax.set_yscale('log')
    ax.set_xlabel('Number of Qubits', fontsize=16)
    ax.set_ylabel('Throughput Advantage  (PTSBE / cuda-Q)', fontsize=16)
    ax.set_title('PTSBE vs cuda-Q \u2014 Per-Config Throughput Advantage',
                 fontsize=17, pad=14)

    ax.set_xticks(QUBITS)
    ax.set_xticklabels([str(q) for q in QUBITS], fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which='major', ls='--', alpha=0.4)

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    for val, label in [(1e3, '1 K\u00d7'), (1e6, '1 M\u00d7')]:
        ax.axhline(y=val, color='#888888', ls='--', lw=0.8, alpha=0.6, zorder=1)
        ax.text(1.02, val, label, transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=10, color='#333333',
                fontweight='bold', clip_on=False,
                bbox=dict(facecolor='white', edgecolor='#999999',
                          boxstyle='round,pad=0.25', linewidth=0.6, alpha=0.9))

    leg = ax.legend(title='Gate Count', fontsize=12, title_fontsize=13,
                    framealpha=0.95, loc='best', fancybox=False,
                    edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    fig.text(0.5, 0.005, FOOTNOTE, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.subplots_adjust(bottom=0.14)

    out = os.path.join(DATA_DIR, 'final_throughput_advantage_stddev_plot.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _plot_200q_vs_gates(adv):
    fig, ax = plt.subplots(figsize=(10, 7.5))

    color_main = '#EE6677'
    color_light = '#EE667755'
    color_annot = '#99334D'

    xs, means, lo, hi = [], [], [], []
    for g in GATES:
        ratios = adv.get((200, g))
        if not ratios:
            continue
        arr = np.array(ratios)
        geo, lo_e, hi_e = _geo_stats(arr)
        xs.append(g)
        means.append(geo)
        lo.append(lo_e)
        hi.append(hi_e)

    if not xs:
        print('No 200q cuda-Q data available \u2014 skipping')
        return

    ax.errorbar(xs, means, yerr=[lo, hi],
                fmt='D-', color=color_main, capsize=6, capthick=1.2,
                markersize=8, linewidth=2.0, elinewidth=1.0,
                markeredgecolor=color_main, markeredgewidth=0.8,
                markerfacecolor=color_main, ecolor=color_light,
                zorder=3)

    for xi, yi in zip(xs, means):
        def fmt(v):
            if v >= 1e6:
                return f'{v/1e6:.1f}M\u00d7'
            if v >= 1e3:
                return f'{v/1e3:.0f}K\u00d7'
            return f'{v:.0f}\u00d7'
        ax.annotate(fmt(yi), (xi, yi), textcoords='offset points',
                    xytext=(0, 14), ha='center', fontsize=11,
                    fontweight='bold', color=color_annot)

    ax.set_yscale('log')
    ax.set_xlabel(r'Number of Gates ($g$)', fontsize=16)
    ax.set_ylabel('Throughput Advantage  (PTSBE / cuda-Q)', fontsize=16)
    ax.set_title('200-Qubit Circuits \u2014 PTSBE vs cuda-Q Throughput Advantage',
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
                              boxstyle='round,pad=0.25', linewidth=0.6,
                              alpha=0.9))

    fig.text(0.5, 0.005, FOOTNOTE, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.subplots_adjust(bottom=0.14)

    out = os.path.join(DATA_DIR, 'final_throughput_advantage_200q_vs_gates.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _plot_gates_axis(adv):
    """X-axis = gate count, one series per qubit count."""
    fig, ax = plt.subplots(figsize=(11, 7.5))

    markers = {50: 'o', 75: 's', 100: '^', 150: 'D', 200: 'v'}
    offsets = {50: -12, 75: -6, 100: 0, 150: 6, 200: 12}

    for q in QUBITS:
        xs, means, lo, hi = [], [], [], []
        for g in GATES:
            ratios = adv.get((q, g))
            if not ratios:
                continue
            arr = np.array(ratios)
            geo, lo_e, hi_e = _geo_stats(arr)
            xs.append(g)
            means.append(geo)
            lo.append(lo_e)
            hi.append(hi_e)

        if not xs:
            continue

        x = np.array(xs, dtype=float) + offsets[q]
        ax.errorbar(x, means, yerr=[lo, hi],
                    fmt=markers[q] + '-', color=COLORS_Q[q],
                    capsize=4, capthick=1.0, markersize=8,
                    linewidth=1.6, elinewidth=1.0,
                    markeredgecolor=COLORS_Q[q], markeredgewidth=0.8,
                    ecolor=COLORS_Q_LIGHT[q],
                    label=r'$n$ = ' + str(q), zorder=3)

    ax.set_yscale('log')
    ax.set_xlabel(r'Number of Gates ($g$)', fontsize=16)
    ax.set_ylabel('Throughput Advantage  (PTSBE / cuda-Q)', fontsize=16)
    ax.set_title('PTSBE vs cuda-Q \u2014 Per-Config Throughput Advantage',
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
        ax.axhline(y=val, color='#888888', ls='--', lw=0.8, alpha=0.6, zorder=1)
        ax.text(1.02, val, label, transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=10, color='#333333',
                fontweight='bold', clip_on=False,
                bbox=dict(facecolor='white', edgecolor='#999999',
                          boxstyle='round,pad=0.25', linewidth=0.6, alpha=0.9))

    leg = ax.legend(title=r'Qubits ($n$)', fontsize=12, title_fontsize=13,
                    framealpha=0.95, loc='best', fancybox=False,
                    edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    fig.text(0.5, 0.005, FOOTNOTE, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.subplots_adjust(bottom=0.14)

    out = os.path.join(DATA_DIR, 'final_throughput_advantage_gates_axis.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _ptsbe_success_rates():
    """Return {(q, g): success_fraction} based on PTSBE log parsing."""
    rates = {}
    for q in QUBITS:
        for g in GATES:
            pf = os.path.join(DATA_DIR, f'{q}q_{g}g_100hs_10nfbs_28fbs_ptsbe.txt')
            if not os.path.isfile(pf):
                continue
            attempted = 0
            succeeded = 0
            in_run = False
            run_aborted = False
            run_has_timing = False
            with open(pf) as f:
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
            rates[(q, g)] = succeeded / max(attempted, 1)
    return rates


def _plot_gates_axis_paper(adv):
    """Paper version of gates-axis plot — NVIDIA palette, hollow markers for <80% PTSBE success."""
    PAPER_GATES = [200, 400, 600, 800, 1000, 1200]

    paper_colors = {
        50:  '#003F72',
        75:  '#0071C5',
        100: '#7B8894',
        150: '#76B900',
        200: '#B4D43C',
    }
    paper_colors_light = {q: c + '55' for q, c in paper_colors.items()}

    success = _ptsbe_success_rates()
    hollow = {(q, g) for (q, g), rate in success.items() if rate < 0.8}

    fig, ax = plt.subplots(figsize=(11, 7.5))

    markers = {50: 'o', 75: 's', 100: '^', 150: 'D', 200: 'v'}
    offsets = {50: -12, 75: -6, 100: 0, 150: 6, 200: 12}

    PAPER_FOOTNOTE = (
        "Markers = geometric mean, error bars = \u00b11 geometric std deviation  |  "
        "10 random circuits per config  |  100 hypersamples\n"
        "Non-final batch size = 10, final batch size = 28  |  "
        "y-axis log scale  |  "
        "Hollow markers = <80% of PTSBE runs succeeded (advantage computed only from mutually successful circuits)"
    )

    plotted_qubits = []
    for q in QUBITS:
        xs, means, lo, hi, gs = [], [], [], [], []
        for g in PAPER_GATES:
            ratios = adv.get((q, g))
            if not ratios:
                continue
            if success.get((q, g), 0) == 0:
                continue
            arr = np.array(ratios)
            geo, lo_e, hi_e = _geo_stats(arr)
            xs.append(g)
            means.append(geo)
            lo.append(lo_e)
            hi.append(hi_e)
            gs.append(g)

        if not xs:
            continue

        plotted_qubits.append(q)
        x = np.array(xs, dtype=float) + offsets[q]
        ax.errorbar(x, means, yerr=[lo, hi],
                    fmt='-', color=paper_colors[q],
                    capsize=4, capthick=1.0,
                    linewidth=1.6, elinewidth=1.0,
                    ecolor=paper_colors_light[q],
                    zorder=3)

        for xi, yi, gi in zip(x, means, gs):
            is_hollow = (q, gi) in hollow
            ax.plot(xi, yi, marker=markers[q], markersize=8,
                    color=paper_colors[q],
                    markeredgecolor=paper_colors[q], markeredgewidth=1.2,
                    markerfacecolor='white' if is_hollow else paper_colors[q],
                    zorder=4)

    ax.set_yscale('log')
    ax.set_xlabel(r'Number of Gates ($g$)', fontsize=16)
    ax.set_ylabel('Throughput Advantage  (PTSBE / cuda-Q)', fontsize=16)
    ax.set_title('PTSBE vs cuda-Q \u2014 Per-Config Throughput Advantage',
                 fontsize=17, pad=14)

    ax.set_xticks(PAPER_GATES)
    ax.set_xticklabels([str(g) for g in PAPER_GATES], fontsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.minorticks_on()
    ax.grid(True, which='major', ls='--', alpha=0.4)

    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)

    for val, label in [(1e3, '1 K\u00d7'), (1e6, '1 M\u00d7')]:
        ax.axhline(y=val, color='#888888', ls='--', lw=0.8, alpha=0.6, zorder=1)
        ax.text(1.02, val, label, transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=10, color='#333333',
                fontweight='bold', clip_on=False,
                bbox=dict(facecolor='white', edgecolor='#999999',
                          boxstyle='round,pad=0.25', linewidth=0.6, alpha=0.9))

    legend_handles = []
    for q in plotted_qubits:
        legend_handles.append(Line2D(
            [0], [0], color=paper_colors[q], marker=markers[q],
            markersize=8, linewidth=1.6,
            markeredgecolor=paper_colors[q], markeredgewidth=1.2,
            markerfacecolor=paper_colors[q], label=r'$n$ = ' + str(q)))
    if hollow:
        legend_handles.append(Line2D(
            [0], [0], color='gray', marker='o', markersize=8,
            linewidth=0, markeredgecolor='gray', markeredgewidth=1.2,
            markerfacecolor='white', label='< 80% success'))
    leg = ax.legend(handles=legend_handles, title=r'Qubits ($n$)',
                    fontsize=12, title_fontsize=13,
                    framealpha=0.95, loc='best', fancybox=False,
                    edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    fig.text(0.5, 0.005, PAPER_FOOTNOTE, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.subplots_adjust(bottom=0.16)

    out = os.path.join(DATA_DIR, 'paper_throughput_advantage_gates_axis.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _blend_hex(hex_color, alpha, bg=(1, 1, 1)):
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (r * alpha + bg[0] * (1 - alpha),
            g * alpha + bg[1] * (1 - alpha),
            b * alpha + bg[2] * (1 - alpha))


def _plot_gates_axis_paper_v2(adv, include_title=True):
    PAPER_GATES = [200, 400, 600, 800, 1000, 1200]

    paper_colors = {
        50:  '#003F72',
        75:  '#0071C5',
        100: '#7B8894',
        150: '#76B900',
        200: '#B4D43C',
    }

    success = _ptsbe_success_rates()
    hollow = {(q, g) for (q, g), rate in success.items() if rate < 0.8}

    markers = {50: 'o', 75: 's', 100: '^', 150: 'D', 200: 'v'}
    offsets = {50: -12, 75: -6, 100: 0, 150: 6, 200: 12}

    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(13, 9))

        plotted_qubits = []
        for q in QUBITS:
            xs, means, lo, hi, gs = [], [], [], [], []
            for g in PAPER_GATES:
                ratios = adv.get((q, g))
                if not ratios:
                    continue
                if success.get((q, g), 0) == 0:
                    continue
                arr = np.array(ratios)
                geo, lo_e, hi_e = _geo_stats(arr)
                xs.append(g)
                means.append(geo)
                lo.append(lo_e)
                hi.append(hi_e)
                gs.append(g)

            if not xs:
                continue

            plotted_qubits.append(q)
            x = np.array(xs, dtype=float) + offsets[q]
            ebar_col = _blend_hex(paper_colors[q], 0.45)
            ax.errorbar(x, means, yerr=[lo, hi],
                        fmt='-', color=paper_colors[q],
                        capsize=5, capthick=2.0,
                        linewidth=3.6, elinewidth=3.6,
                        ecolor=ebar_col, zorder=3)

            for xi, yi, gi in zip(x, means, gs):
                is_hollow = (q, gi) in hollow
                ax.plot(xi, yi, marker=markers[q], markersize=13,
                        color=paper_colors[q],
                        markeredgecolor=paper_colors[q], markeredgewidth=1.2,
                        markerfacecolor='white' if is_hollow else paper_colors[q],
                        zorder=4)

        ax.set_yscale('log')
        ax.set_xlabel(r'Number of Gates ($g$)', fontsize=28)
        ax.set_ylabel('Data Collection Speedup  (PTSBE / cuda-Q)', fontsize=28)
        if include_title:
            ax.set_title('PTSBE vs cuda-Q \u2014 Per-Config Data Collection Speedup',
                         fontsize=28, pad=14)

        ax.set_xticks(PAPER_GATES)
        ax.set_xticklabels([str(g) for g in PAPER_GATES], fontsize=23)
        ax.tick_params(axis='y', labelsize=26)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        legend_handles = []
        for q in plotted_qubits:
            legend_handles.append(Line2D(
                [0], [0], color=paper_colors[q], marker=markers[q],
                markersize=13, linewidth=3.6,
                markeredgecolor=paper_colors[q], markeredgewidth=1.2,
                markerfacecolor=paper_colors[q], label=r'$n$ = ' + str(q)))
        if hollow:
            legend_handles.append(Line2D(
                [0], [0], color='gray', marker='o', markersize=10,
                linewidth=0, markeredgecolor='gray', markeredgewidth=1.2,
                markerfacecolor='white', label='< 80% success'))
        leg = ax.legend(handles=legend_handles, title=r'Qubits ($n$)',
                        fontsize=23, title_fontsize=24,
                        framealpha=1.0, loc='best', fancybox=False,
                        edgecolor='black')
        leg.get_frame().set_linewidth(0.6)

        fig.tight_layout(pad=1.5)

        suffix = 'v2' if include_title else 'v3'
        out_pdf = os.path.join(DATA_DIR, f'paper_throughput_advantage_gates_axis_{suffix}.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(DATA_DIR, f'paper_throughput_advantage_gates_axis_{suffix}.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_png}')

        plt.close(fig)


def main():
    adv = _collect_advantages()

    print('Throughput advantage summary (PTSBE / cuda-Q, precise cudaq):')
    print(f'{"Config":<16} {"n":>3}  {"GeoMean":>12}  {"Min":>12}  {"Max":>12}')
    print('-' * 60)
    for (q, g), ratios in sorted(adv.items()):
        if not ratios:
            continue
        a = np.array(ratios)
        geo = np.exp(np.mean(np.log(a)))
        print(f'{q}q_{g}g{"":<8} {len(a):>3}  {geo:>12,.0f}×  '
              f'{np.min(a):>12,.0f}×  {np.max(a):>12,.0f}×')

    _plot_gates_axis_paper_v4(adv)


def _plot_gates_axis_paper_v4(adv):
    """V4: larger fonts across all elements."""
    PAPER_GATES = [200, 400, 600, 800, 1000, 1200]
    paper_colors = {50: '#003F72', 75: '#0071C5', 100: '#7B8894',
                    150: '#76B900', 200: '#B4D43C'}
    success = _ptsbe_success_rates()
    hollow = {(q, g) for (q, g), rate in success.items() if rate < 0.8}

    markers = {50: 'o', 75: 's', 100: '^', 150: 'D', 200: 'v'}
    offsets = {50: -12, 75: -6, 100: 0, 150: 6, 200: 12}

    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(13, 9))

        plotted_qubits = []
        for q in QUBITS:
            xs, means, lo, hi, gs = [], [], [], [], []
            for g in PAPER_GATES:
                ratios = adv.get((q, g))
                if not ratios:
                    continue
                if success.get((q, g), 0) == 0:
                    continue
                arr = np.array(ratios)
                geo, lo_e, hi_e = _geo_stats(arr)
                xs.append(g)
                means.append(geo)
                lo.append(lo_e)
                hi.append(hi_e)
                gs.append(g)

            if not xs:
                continue

            plotted_qubits.append(q)
            x = np.array(xs, dtype=float) + offsets[q]
            ebar_col = _blend_hex(paper_colors[q], 0.45)
            ax.errorbar(x, means, yerr=[lo, hi],
                        fmt='-', color=paper_colors[q],
                        capsize=5, capthick=2.0,
                        linewidth=3.6, elinewidth=3.6,
                        ecolor=ebar_col, zorder=3)

            for xi, yi, gi in zip(x, means, gs):
                is_hollow = (q, gi) in hollow
                ax.plot(xi, yi, marker=markers[q], markersize=15,
                        color=paper_colors[q],
                        markeredgecolor=paper_colors[q], markeredgewidth=1.4,
                        markerfacecolor='white' if is_hollow else paper_colors[q],
                        zorder=4)

        ax.set_yscale('log')
        ax.set_xlabel(r'Number of Gates ($g$)', fontsize=32)
        ax.set_ylabel('Data Collection Speedup  (PTSBE / cuda-Q)', fontsize=32)

        ax.set_xticks(PAPER_GATES)
        ax.set_xticklabels([str(g) for g in PAPER_GATES], fontsize=27)
        ax.tick_params(axis='y', labelsize=28)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        legend_handles = []
        for q in plotted_qubits:
            legend_handles.append(Line2D(
                [0], [0], color=paper_colors[q], marker=markers[q],
                markersize=15, linewidth=3.6,
                markeredgecolor=paper_colors[q], markeredgewidth=1.4,
                markerfacecolor=paper_colors[q], label=r'$n$ = ' + str(q)))
        if hollow:
            legend_handles.append(Line2D(
                [0], [0], color='gray', marker='o', markersize=12,
                linewidth=0, markeredgecolor='gray', markeredgewidth=1.4,
                markerfacecolor='white', label='< 80% success'))
        leg = ax.legend(handles=legend_handles, title=r'Qubits ($n$)',
                        fontsize=26, title_fontsize=27,
                        framealpha=1.0, loc='best', fancybox=False,
                        edgecolor='black')
        leg.get_frame().set_linewidth(0.6)

        fig.tight_layout(pad=1.5)

        out_pdf = os.path.join(DATA_DIR, 'paper_throughput_advantage_gates_axis_v4.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(DATA_DIR, 'paper_throughput_advantage_gates_axis_v4.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_png}')

        plt.close(fig)


if __name__ == '__main__':
    main()
