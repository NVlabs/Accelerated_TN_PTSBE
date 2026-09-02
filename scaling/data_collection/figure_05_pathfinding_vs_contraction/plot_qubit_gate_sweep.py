#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Presentation-quality plots for the Figure 5 qubit × gate sweep.

Qubits=[50,75,100,150,200], Gates=[200,400,600,800,1000], Hypersamples=1.
Batch config fixed: nfbs=10, fbs=28, 1 noise sample, 10 repeats.

Generates:
  final_contraction_time_panel.png  — side-by-side: contraction time (x=gates, legend=qubits)
                                       and contraction time (x=qubits, legend=gates)
  final_pathfinding_time_panel.png  — same layout for path-finding time
  final_combined_panel.png          — 2-panel: contraction + pathfinding, x=gates, legend=qubits
"""

import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

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
GATES = [200, 400, 600, 800, 1000]
HS = 1

COLORS_Q = {50: '#4477AA', 75: '#66CCEE', 100: '#228833', 150: '#CCBB44', 200: '#EE6677'}
MARKERS_Q = {50: 'o', 75: 'v', 100: 's', 150: '^', 200: 'D'}

COLORS_G = {200: '#4477AA', 400: '#66CCEE', 600: '#228833', 800: '#CCBB44', 1000: '#EE6677'}
MARKERS_G = {200: 'o', 400: 'v', 600: 's', 800: '^', 1000: 'D'}

FOOTNOTE = (
    "Markers = geometric mean, error bars = +/-1 geometric std deviation  |  "
    "10 random circuits x 10 repeats per config  |  1 noise sample, 1 hypersample\n"
    "Non-final batch size = 10, final batch size = 28  |  "
    "y-axis log scale"
)


def _extract_timings(filepath):
    """Extract (contraction_loop_time, pathfinding_time) pairs from RUN 1/N only."""
    if not os.path.isfile(filepath):
        return []

    results = []
    contraction_time = None
    pathfinding_time = None
    is_run1 = False

    with open(filepath) as f:
        for line in f:
            run_match = re.search(r'RUN (\d+)/\d+ \| circuit_id=(\d+)', line)
            if run_match:
                if is_run1 and contraction_time is not None and pathfinding_time is not None:
                    results.append((contraction_time, pathfinding_time))
                contraction_time = None
                pathfinding_time = None
                is_run1 = (int(run_match.group(1)) == 1)

            if not is_run1:
                continue

            mc = re.search(r'time_contraction_loop\s*:\s*([\d.]+)s', line)
            if mc:
                contraction_time = float(mc.group(1))

            mp = re.search(r'time_build_expr_operands\s*:\s*([\d.]+)s', line)
            if mp:
                pathfinding_time = float(mp.group(1))

    if is_run1 and contraction_time is not None and pathfinding_time is not None:
        results.append((contraction_time, pathfinding_time))

    return results


def _collect_data():
    """Returns dicts keyed by (qubits, gates) -> list of timing values."""
    contraction_data = defaultdict(list)
    pathfinding_data = defaultdict(list)

    for q in QUBITS:
        for g in GATES:
            fname = f'{q}q_{g}g_1hs_10nfbs_28fbs_ptsbe.txt'
            fpath = os.path.join(DATA_DIR, fname)
            timings = _extract_timings(fpath)
            for ct, pf in timings:
                if ct > 0:
                    contraction_data[(q, g)].append(ct)
                if pf > 0:
                    pathfinding_data[(q, g)].append(pf)

    return contraction_data, pathfinding_data


def _extract_timings_per_shot(filepath):
    """Extract matched per-shot timings and their ratio from RUN 1/N."""
    if not os.path.isfile(filepath):
        return []

    results = []
    contraction_time = None
    pathfinding_time = None
    total_shots = None
    is_run1 = False

    with open(filepath) as f:
        for line in f:
            run_match = re.search(r'RUN (\d+)/\d+ \| circuit_id=(\d+)', line)
            if run_match:
                if is_run1 and contraction_time is not None and pathfinding_time is not None and total_shots and total_shots > 0:
                    ct_per_shot = contraction_time / total_shots
                    results.append((ct_per_shot, pathfinding_time / total_shots,
                                    pathfinding_time / ct_per_shot))
                contraction_time = None
                pathfinding_time = None
                total_shots = None
                is_run1 = (int(run_match.group(1)) == 1)

            if not is_run1:
                continue

            mc = re.search(r'time_contraction_loop\s*:\s*([\d.]+)s', line)
            if mc:
                contraction_time = float(mc.group(1))

            mp = re.search(r'time_build_expr_operands\s*:\s*([\d.]+)s', line)
            if mp:
                pathfinding_time = float(mp.group(1))

            ms = re.search(r'Total number of overall PTSBE shots collected:\s+(\d+)', line)
            if ms:
                total_shots = int(ms.group(1))

    if is_run1 and contraction_time is not None and pathfinding_time is not None and total_shots and total_shots > 0:
        ct_per_shot = contraction_time / total_shots
        results.append((ct_per_shot, pathfinding_time / total_shots,
                        pathfinding_time / ct_per_shot))

    return results


def _collect_data_per_shot():
    """Return matched per-shot timings and ratios keyed by (qubits, gates)."""
    contraction_data = defaultdict(list)
    pathfinding_data = defaultdict(list)
    ratio_data = defaultdict(list)

    for q in QUBITS:
        for g in GATES:
            fname = f'{q}q_{g}g_1hs_10nfbs_28fbs_ptsbe.txt'
            fpath = os.path.join(DATA_DIR, fname)
            timings = _extract_timings_per_shot(fpath)
            for ct, pf, ratio in timings:
                if ct > 0:
                    contraction_data[(q, g)].append(ct)
                if pf > 0:
                    pathfinding_data[(q, g)].append(pf)
                if ratio > 0:
                    ratio_data[(q, g)].append(ratio)

    return contraction_data, pathfinding_data, ratio_data


def _geo_stats(arr):
    log_arr = np.log(arr)
    geo_mean = np.exp(np.mean(log_arr))
    geo_std = np.exp(np.std(log_arr))
    return geo_mean, geo_mean - geo_mean / geo_std, geo_mean * geo_std - geo_mean


def _make_panel_gates_axis(contraction_data, pathfinding_data):
    """2-panel plot: contraction + pathfinding, x=gates, legend=qubits."""
    fig, (ax_ct, ax_pf) = plt.subplots(1, 2, figsize=(20, 8))
    offsets = {50: -16, 75: -8, 100: 0, 150: 8, 200: 16}

    for ax, data, ylabel in [
        (ax_ct, contraction_data, 'Contraction Loop Time (s)'),
        (ax_pf, pathfinding_data, 'Path-Finding Time (s)'),
    ]:
        for q in QUBITS:
            xs, means, lo, hi = [], [], [], []
            for g in GATES:
                vals = data.get((q, g))
                if not vals or len(vals) < 3:
                    continue
                arr = np.array(vals)
                geo, lo_e, hi_e = _geo_stats(arr)
                xs.append(g)
                means.append(geo)
                lo.append(lo_e)
                hi.append(hi_e)

            if not xs:
                continue

            x = np.array(xs, dtype=float) + offsets[q]
            ax.errorbar(x, means, yerr=[lo, hi],
                        fmt=MARKERS_Q[q] + '-', color=COLORS_Q[q],
                        capsize=4, capthick=1.0, markersize=8,
                        linewidth=1.6, elinewidth=1.0,
                        markeredgecolor=COLORS_Q[q], markeredgewidth=0.8,
                        ecolor=COLORS_Q[q] + '55',
                        label=r'$n$ = ' + str(q), zorder=3)

        ax.set_yscale('log')
        ax.set_xlabel(r'Number of Gates ($g$)', fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xticks(GATES)
        ax.set_xticklabels([str(g) for g in GATES], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        ax.grid(True, which='major', ls='--', alpha=0.4)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    ax_ct.set_title('Contraction Time', fontsize=16, pad=10)
    ax_pf.set_title('Path-Finding Time', fontsize=16, pad=10)

    leg = ax_pf.legend(title='Qubits', fontsize=11, title_fontsize=12,
                       framealpha=0.95, loc='best', fancybox=False,
                       edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    fig.suptitle('Contraction Time & Path-Finding Time vs Gates (1 Hypersample)',
                 fontsize=18, y=1.01)
    fig.text(0.5, -0.01, FOOTNOTE, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.tight_layout()

    out = os.path.join(DATA_DIR, 'final_combined_panel.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _make_panel_qubits_axis(contraction_data, pathfinding_data):
    """2-panel plot: contraction + pathfinding, x=qubits, legend=gates."""
    fig, (ax_ct, ax_pf) = plt.subplots(1, 2, figsize=(20, 8))
    offsets = {200: -6, 400: -3, 600: 0, 800: 3, 1000: 6}

    for ax, data, ylabel in [
        (ax_ct, contraction_data, 'Contraction Loop Time (s)'),
        (ax_pf, pathfinding_data, 'Path-Finding Time (s)'),
    ]:
        for g in GATES:
            xs, means, lo, hi = [], [], [], []
            for q in QUBITS:
                vals = data.get((q, g))
                if not vals or len(vals) < 3:
                    continue
                arr = np.array(vals)
                geo, lo_e, hi_e = _geo_stats(arr)
                xs.append(q)
                means.append(geo)
                lo.append(lo_e)
                hi.append(hi_e)

            if not xs:
                continue

            x = np.array(xs, dtype=float) + offsets[g]
            ax.errorbar(x, means, yerr=[lo, hi],
                        fmt=MARKERS_G[g] + '-', color=COLORS_G[g],
                        capsize=4, capthick=1.0, markersize=8,
                        linewidth=1.6, elinewidth=1.0,
                        markeredgecolor=COLORS_G[g], markeredgewidth=0.8,
                        ecolor=COLORS_G[g] + '55',
                        label=f'{g} gates', zorder=3)

        ax.set_yscale('log')
        ax.set_xlabel('Number of Qubits', fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xticks(QUBITS)
        ax.set_xticklabels([str(q) for q in QUBITS], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        ax.grid(True, which='major', ls='--', alpha=0.4)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    ax_ct.set_title('Contraction Time', fontsize=16, pad=10)
    ax_pf.set_title('Path-Finding Time', fontsize=16, pad=10)

    leg = ax_pf.legend(title='Gates', fontsize=11, title_fontsize=12,
                       framealpha=0.95, loc='best', fancybox=False,
                       edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    fig.suptitle('Contraction Time & Path-Finding Time vs Qubits (1 Hypersample)',
                 fontsize=18, y=1.01)
    fig.text(0.5, -0.01, FOOTNOTE, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.tight_layout()

    out = os.path.join(DATA_DIR, 'final_combined_panel_qubits_axis.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _hollow_set(contraction_data, expected=10, threshold=0.8):
    """Return set of (q, g) configs with success rate < threshold."""
    hollow = set()
    for (q, g), vals in contraction_data.items():
        if len(vals) / expected < threshold:
            hollow.add((q, g))
    return hollow


def _plot_errorbar_with_hollow(ax, xs_raw, means, lo, hi, gates_list,
                               q, offsets, paper_colors, hollow, is_first_ax):
    """Plot errorbar lines + markers, using hollow markers for low-success configs."""
    x = np.array(xs_raw, dtype=float) + offsets[q]

    ax.errorbar(x, means, yerr=[lo, hi],
                fmt='-', color=paper_colors[q],
                capsize=4, capthick=1.0, markersize=0,
                linewidth=1.6, elinewidth=1.0,
                ecolor=paper_colors[q] + '55',
                label=(r'$n$ = ' + str(q)) if is_first_ax else None, zorder=3)

    for i, g in enumerate(gates_list):
        fc = 'white' if (q, g) in hollow else paper_colors[q]
        ax.plot(x[i], means[i], MARKERS_Q[q], color=paper_colors[q],
                markersize=8, markeredgewidth=1.2,
                markeredgecolor=paper_colors[q], markerfacecolor=fc,
                zorder=4)


def _make_paper_3panel(contraction_data, pathfinding_data,
                       ratio_numerator, ratio_denominator,
                       ratio_label, ratio_title, out_filename):
    """Paper-quality 3-panel plot (x=gates, legend=qubits) with NVIDIA palette.
    Hollow markers for configs with <80% success rate."""
    paper_colors = {50: '#003F72', 75: '#0071C5', 100: '#7B8894',
                    150: '#76B900', 200: '#B4D43C'}

    hollow = _hollow_set(contraction_data)

    fig, (ax_ct, ax_pf, ax_ratio) = plt.subplots(1, 3, figsize=(28, 8))
    offsets = {50: -16, 75: -8, 100: 0, 150: 8, 200: 16}

    first_ax = True
    for ax, data, ylabel in [
        (ax_ct, contraction_data, 'Contraction Loop Time (s)'),
        (ax_pf, pathfinding_data, 'Path-Finding Time (s)'),
    ]:
        for q in QUBITS:
            xs_gates, means, lo, hi = [], [], [], []
            for g in GATES:
                vals = data.get((q, g))
                if not vals or len(vals) < 3:
                    continue
                arr = np.array(vals)
                geo, lo_e, hi_e = _geo_stats(arr)
                xs_gates.append(g)
                means.append(geo)
                lo.append(lo_e)
                hi.append(hi_e)

            if not xs_gates:
                continue

            _plot_errorbar_with_hollow(ax, xs_gates, means, lo, hi, xs_gates,
                                       q, offsets, paper_colors, hollow, first_ax)

        ax.set_yscale('log')
        ax.set_xlabel(r'Number of Gates ($g$)', fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xticks(GATES)
        ax.set_xticklabels([str(g) for g in GATES], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        ax.grid(True, which='major', ls='--', alpha=0.4)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        first_ax = False

    ax_ct.set_title('Contraction Time', fontsize=16, pad=10)
    ax_pf.set_title('Path-Finding Time', fontsize=16, pad=10)

    num_data = contraction_data if ratio_numerator == 'contraction' else pathfinding_data
    den_data = pathfinding_data if ratio_numerator == 'contraction' else contraction_data

    for q in QUBITS:
        xs_gates, means, lo, hi = [], [], [], []
        for g in GATES:
            num_vals = num_data.get((q, g))
            den_vals = den_data.get((q, g))
            if not num_vals or not den_vals or len(num_vals) < 3:
                continue
            n = min(len(num_vals), len(den_vals))
            ratios = np.array(num_vals[:n]) / np.array(den_vals[:n])
            geo, lo_e, hi_e = _geo_stats(ratios)
            xs_gates.append(g)
            means.append(geo)
            lo.append(lo_e)
            hi.append(hi_e)

        if not xs_gates:
            continue

        _plot_errorbar_with_hollow(ax_ratio, xs_gates, means, lo, hi, xs_gates,
                                   q, offsets, paper_colors, hollow, False)

    ax_ratio.set_yscale('log')
    ax_ratio.set_xlabel('Number of Gates', fontsize=15)
    ax_ratio.set_ylabel(ratio_label, fontsize=15)
    ax_ratio.set_title(ratio_title, fontsize=16, pad=10)
    ax_ratio.set_xticks(GATES)
    ax_ratio.set_xticklabels([str(g) for g in GATES], fontsize=12)
    ax_ratio.tick_params(axis='y', labelsize=12)
    ax_ratio.tick_params(which='both', direction='in', top=True, right=True)
    ax_ratio.minorticks_on()
    ax_ratio.grid(True, which='major', ls='--', alpha=0.4)
    ax_ratio.axhline(y=1.0, color='#333333', linewidth=1.2, linestyle='--', alpha=0.6, zorder=1)
    for spine in ('top', 'right'):
        ax_ratio.spines[spine].set_visible(False)

    plotted_qubits = [q for q in QUBITS
                      if any(len(contraction_data.get((q, g), [])) >= 3 for g in GATES)]
    legend_handles = []
    for q in plotted_qubits:
        legend_handles.append(Line2D(
            [0], [0], color=paper_colors[q], marker=MARKERS_Q[q],
            markersize=8, linewidth=1.6,
            markeredgecolor=paper_colors[q], markeredgewidth=1.2,
            markerfacecolor=paper_colors[q], label=r'$n$ = ' + str(q)))
    if hollow:
        legend_handles.append(Line2D(
            [0], [0], color='gray', marker='o', markersize=8,
            linewidth=0, markeredgecolor='gray', markeredgewidth=1.2,
            markerfacecolor='white', label='< 80% success'))
    fig.legend(handles=legend_handles, loc='upper center',
               ncol=len(legend_handles),
               fontsize=12, framealpha=0.95, fancybox=False,
               edgecolor='black', bbox_to_anchor=(0.5, 0.99),
               columnspacing=2.0, handletextpad=0.6)

    fig.suptitle('PTSBE Computational Cost Breakdown:\nPath-Finding vs Contraction Across Qubit-Gate Configurations',
                 fontsize=18, y=1.06)

    hollow_footnote = FOOTNOTE + '\nHollow markers = <80% of PTSBE runs succeeded'
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.5, 0.005, hollow_footnote, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)

    out = os.path.join(DATA_DIR, out_filename)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved  {out}')
    plt.close(fig)


def _make_panel_qubits_axis_paper(contraction_data, pathfinding_data):
    """Paper-quality 2-panel plot (x=qubits, legend=gates) with NVIDIA palette."""
    paper_colors = {200: '#003F72', 400: '#0071C5', 600: '#7B8894',
                    800: '#76B900', 1000: '#B4D43C'}

    fig, (ax_ct, ax_pf) = plt.subplots(1, 2, figsize=(20, 8))
    offsets = {200: -6, 400: -3, 600: 0, 800: 3, 1000: 6}

    for ax, data, ylabel in [
        (ax_ct, contraction_data, 'Contraction Loop Time (s)'),
        (ax_pf, pathfinding_data, 'Path-Finding Time (s)'),
    ]:
        for g in GATES:
            xs, means, lo, hi = [], [], [], []
            for q in QUBITS:
                vals = data.get((q, g))
                if not vals or len(vals) < 3:
                    continue
                arr = np.array(vals)
                geo, lo_e, hi_e = _geo_stats(arr)
                xs.append(q)
                means.append(geo)
                lo.append(lo_e)
                hi.append(hi_e)

            if not xs:
                continue

            x = np.array(xs, dtype=float) + offsets[g]
            ax.errorbar(x, means, yerr=[lo, hi],
                        fmt=MARKERS_G[g] + '-', color=paper_colors[g],
                        capsize=4, capthick=1.0, markersize=8,
                        linewidth=1.6, elinewidth=1.0,
                        markeredgecolor=paper_colors[g], markeredgewidth=0.8,
                        ecolor=paper_colors[g] + '55',
                        label=f'{g} gates', zorder=3)

        ax.set_yscale('log')
        ax.set_xlabel('Number of Qubits', fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.set_xticks(QUBITS)
        ax.set_xticklabels([str(q) for q in QUBITS], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        ax.grid(True, which='major', ls='--', alpha=0.4)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    ax_ct.set_title('Contraction Time', fontsize=16, pad=10)
    ax_pf.set_title('Path-Finding Time', fontsize=16, pad=10)

    leg = ax_pf.legend(title='Gates', fontsize=11, title_fontsize=12,
                       framealpha=0.95, loc='best', fancybox=False,
                       edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    fig.suptitle('Contraction Time & Path-Finding Time vs Qubits (1 Hypersample)',
                 fontsize=18, y=1.01)
    fig.text(0.5, -0.01, FOOTNOTE, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.tight_layout()

    out = os.path.join(DATA_DIR, 'paper_combined_panel_qubits_axis.png')
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


def _plot_errorbar_with_hollow_v2(ax, xs_raw, means, lo, hi, gates_list,
                                   q, offsets, paper_colors, hollow, is_first_ax):
    """Plot errorbar with pre-blended colors for vector PDF output."""
    x = np.array(xs_raw, dtype=float) + offsets[q]
    ebar_col = _blend_hex(paper_colors[q], 0.65)

    ax.errorbar(x, means, yerr=[lo, hi],
                fmt='-', color=paper_colors[q],
                capsize=8, capthick=2.0, markersize=0,
                linewidth=4.0, elinewidth=2.0,
                ecolor=ebar_col,
                label=(r'$n$ = ' + str(q)) if is_first_ax else None, zorder=3)

    for i, g in enumerate(gates_list):
        fc = 'white' if (q, g) in hollow else paper_colors[q]
        ax.plot(x[i], means[i], MARKERS_Q[q], color=paper_colors[q],
                markersize=23, markeredgewidth=2.5,
                markeredgecolor=paper_colors[q], markerfacecolor=fc,
                zorder=4)


def _make_paper_3panel_v2(contraction_data, pathfinding_data, ratio_data,
                          ratio_label, ratio_title, out_base,
                          include_title=True, font_bump=0,
                          ylabel_ct='Contraction Time (s)', ylabel_pf='Path-Finding Time (s)',
                          overlay_stars=None):
    """V2 paper-quality 3-panel plot: bigger fonts, no footnote, PDF+PNG vector output."""
    paper_colors = {50: '#003F72', 75: '#0071C5', 100: '#7B8894',
                    150: '#76B900', 200: '#B4D43C'}
    hollow = _hollow_set(contraction_data)
    fb = font_bump

    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, (ax_ct, ax_pf, ax_ratio) = plt.subplots(1, 3, figsize=(52, 16))
        offsets = {50: -16, 75: -8, 100: 0, 150: 8, 200: 16}

        first_ax = True
        for ax, data, ylabel in [
            (ax_ct, contraction_data, ''),
            (ax_pf, pathfinding_data, ''),
        ]:
            for q in QUBITS:
                xs_gates, means, lo, hi = [], [], [], []
                for g in GATES:
                    vals = data.get((q, g))
                    if not vals or len(vals) < 3:
                        continue
                    arr = np.array(vals)
                    geo, lo_e, hi_e = _geo_stats(arr)
                    xs_gates.append(g)
                    means.append(geo)
                    lo.append(lo_e)
                    hi.append(hi_e)

                if not xs_gates:
                    continue

                _plot_errorbar_with_hollow_v2(ax, xs_gates, means, lo, hi, xs_gates,
                                              q, offsets, paper_colors, hollow, first_ax)

            ax.set_yscale('log')
            ax.set_xlabel(r'Number of Gates ($g$)', fontsize=59+fb, labelpad=10)
            ax.set_ylabel(ylabel, fontsize=59+fb)
            ax.set_xticks(GATES)
            ax.set_xticklabels([str(g) for g in GATES], fontsize=46+fb)
            ax.tick_params(axis='y', labelsize=46+fb)
            ax.tick_params(which='major', direction='in', top=True, right=True,
                           length=12, width=2, pad=10)
            ax.tick_params(which='minor', direction='in', top=True, right=True,
                           length=6, width=1.5, pad=10)
            ax.minorticks_on()
            grid_col = _blend_hex('#000000', 0.12)
            ax.grid(True, which='major', ls='--', color=grid_col)
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            first_ax = False

        ax_ct.set_title(ylabel_ct, fontsize=59+fb, pad=20)
        ax_pf.set_title(ylabel_pf, fontsize=59+fb, pad=20)

        for q in QUBITS:
            xs_gates, means, lo, hi = [], [], [], []
            for g in GATES:
                ratio_vals = ratio_data.get((q, g))
                if not ratio_vals or len(ratio_vals) < 3:
                    continue
                ratios = np.array(ratio_vals)
                geo, lo_e, hi_e = _geo_stats(ratios)
                xs_gates.append(g)
                means.append(geo)
                lo.append(lo_e)
                hi.append(hi_e)

            if not xs_gates:
                continue

            _plot_errorbar_with_hollow_v2(ax_ratio, xs_gates, means, lo, hi, xs_gates,
                                          q, offsets, paper_colors, hollow, False)

        ax_ratio.set_yscale('log')
        ax_ratio.set_xlabel(r'Number of Gates ($g$)', fontsize=59+fb, labelpad=10)
        ax_ratio.set_ylabel('', fontsize=59+fb)
        ax_ratio.set_title(ratio_title, fontsize=59+fb, pad=20)
        ax_ratio.set_xticks(GATES)
        ax_ratio.set_xticklabels([str(g) for g in GATES], fontsize=46+fb)
        ax_ratio.tick_params(axis='y', labelsize=46+fb)
        ax_ratio.tick_params(which='major', direction='in', top=True, right=True,
                             length=12, width=2, pad=10)
        ax_ratio.tick_params(which='minor', direction='in', top=True, right=True,
                             length=6, width=1.5, pad=10)
        ax_ratio.minorticks_on()
        ax_ratio.grid(True, which='major', ls='--', color=grid_col)
        hline_col = _blend_hex('#333333', 0.6)
        ax_ratio.axhline(y=1.0, color=hline_col, linewidth=2.4, linestyle='--', zorder=1)
        for spine in ('top', 'right'):
            ax_ratio.spines[spine].set_visible(False)

        if overlay_stars:
            star_colors = ['#C04040', '#704030']
            for si, star in enumerate(overlay_stars):
                sc = star_colors[si % len(star_colors)]
                fc = 'white' if star.get('hollow', False) else sc
                g_pos = star['ng']
                for ax, val, lo, hi in [
                    (ax_ct, star['ct_per_shot'], star['ct_lo'], star['ct_hi']),
                    (ax_pf, star['pf'], star['pf_lo'], star['pf_hi']),
                    (ax_ratio, star['ratio'], star['ratio_lo'], star['ratio_hi']),
                ]:
                    ax.errorbar(g_pos, val, yerr=[[lo], [hi]],
                                fmt='none', capsize=8, capthick=2.5,
                                elinewidth=2.5, ecolor=sc, zorder=10)
                    ax.plot(g_pos, val, '*', color=fc, markersize=35+fb,
                            markeredgecolor=sc, markeredgewidth=2.0, zorder=11)

        plotted_qubits = [q for q in QUBITS
                          if any(len(contraction_data.get((q, g), [])) >= 3 for g in GATES)]
        legend_handles = []
        for q in plotted_qubits:
            legend_handles.append(Line2D(
                [0], [0], color=paper_colors[q], marker=MARKERS_Q[q],
                markersize=23+fb, linewidth=4.0+fb,
                markeredgecolor=paper_colors[q], markeredgewidth=2.5,
                markerfacecolor=paper_colors[q], label=r'$n$ = ' + str(q)))
        if hollow:
            legend_handles.append(Line2D(
                [0], [0], color='gray', marker='o', markersize=22,
                linewidth=0, markeredgecolor='gray', markeredgewidth=2.5,
                markerfacecolor='white', label='< 80% success'))
        if overlay_stars:
            star_colors = ['#C04040', '#704030']
            for si, star in enumerate(overlay_stars):
                sc = star_colors[si % len(star_colors)]
                fc = 'white' if star.get('hollow', False) else sc
                legend_handles.append(Line2D(
                    [0], [0], color=fc, marker='*', markersize=28+fb,
                    linewidth=0, markeredgecolor=sc, markeredgewidth=2.0,
                    markerfacecolor=fc, label=star['label']))
        top_rect = 0.72

        fig.tight_layout(rect=[0, 0, 1, top_rect])

        fig.legend(handles=legend_handles, loc='upper center',
                   ncol=len(legend_handles),
                   fontsize=48+fb, framealpha=1.0, fancybox=False,
                   edgecolor='black', bbox_to_anchor=(0.5, 0.88),
                   columnspacing=2.0, handletextpad=0.6)

        if include_title:
            fig.suptitle('PTSBE Computational Cost Breakdown:\n'
                         'Path-Finding vs Contraction Across Qubit-Gate Configurations',
                         fontsize=59, y=0.99)

        out_pdf = os.path.join(DATA_DIR, out_base + '.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(DATA_DIR, out_base + '.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight')
        print(f'Saved  {out_png}')

        plt.close(fig)


def main():
    contraction_data, pathfinding_data = _collect_data()

    if not contraction_data and not pathfinding_data:
        print('No data files found yet. Run the experiments first.')
        return

    print(f'{"Config":<28} {"n":>5}  {"Contraction":>14}  {"PathFinding":>14}')
    print('-' * 70)
    for q in QUBITS:
        for g in GATES:
            ct = contraction_data.get((q, g), [])
            pf = pathfinding_data.get((q, g), [])
            if not ct:
                status = "(aborted)" if os.path.isfile(
                    os.path.join(DATA_DIR, f'{q}q_{g}g_1hs_10nfbs_28fbs_ptsbe.txt')) else "(no data)"
                print(f'{q}q_{g}g{"":<18} {status}')
                continue
            ct_geo = np.exp(np.mean(np.log(np.array(ct))))
            pf_geo = np.exp(np.mean(np.log(np.array(pf)))) if pf else 0
            print(f'{q}q_{g}g{"":<18} {len(ct):>5}  '
                  f'{ct_geo:>13.3f}s  {pf_geo:>13.3f}s')

    ct_ps, _, ratio_data = _collect_data_per_shot()
    _, pf_raw = _collect_data()

    prop3_stars = _collect_proportional3_stars()

    _make_paper_3panel_v2(ct_ps, pf_raw, ratio_data,
                          ratio_label='Path-Finding / Contraction Per Shot',
                          ratio_title='Ratio (Path-Finding / Contraction Per Shot)',
                          out_base='paper2_combined_panel_gates_axis_v4',
                          include_title=False,
                          font_bump=4,
                          ylabel_ct='Contraction Time Per Shot (s)',
                          ylabel_pf='Path-Finding Time (s)',
                          overlay_stars=prop3_stars)


def _collect_proportional3_stars():
    """Collect per-shot contraction time, pathfinding time from proportional3 for 100q/600g and 200q/1000g."""
    PROP3_DIR = os.path.join(DATA_DIR, '..', 'figure_04_panel_b_proportional_speedup')
    NSHOTS_LIST = [10000]
    configs = [(100, 600), (200, 1000)]

    stars = []
    for nq, ng in configs:
        ct_per_shot_all = []
        pf_all = []
        ratio_all = []
        for ns in NSHOTS_LIST:
            fname = f'{nq}q_{ng}g_100hs_10bs_proportional_{ns}nshots_ptsbe.txt'
            fpath = os.path.join(PROP3_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            contraction_time = None
            pathfinding_time = None
            total_shots = None
            with open(fpath) as f:
                for line in f:
                    m = re.search(r'circuit_id=(\d+)', line)
                    if m:
                        if contraction_time is not None and pathfinding_time is not None and total_shots and total_shots > 0:
                            ct_ps = contraction_time / total_shots
                            ct_per_shot_all.append(ct_ps)
                            pf_all.append(pathfinding_time)
                            if ct_ps > 0:
                                ratio_all.append(pathfinding_time / ct_ps)
                        contraction_time = None
                        pathfinding_time = None
                        total_shots = None

                    mc = re.search(r'time_contraction_loop\s*:\s*([\d.]+)s', line)
                    if mc:
                        contraction_time = float(mc.group(1))
                    mp = re.search(r'time_build_expr_operands\s*:\s*([\d.]+)s', line)
                    if mp:
                        pathfinding_time = float(mp.group(1))
                    ms = re.search(r'Total number of overall PTSBE shots collected:\s+(\d+)', line)
                    if ms:
                        total_shots = int(ms.group(1))

                if contraction_time is not None and pathfinding_time is not None and total_shots and total_shots > 0:
                    ct_ps = contraction_time / total_shots
                    ct_per_shot_all.append(ct_ps)
                    pf_all.append(pathfinding_time)
                    if ct_ps > 0:
                        ratio_all.append(pathfinding_time / ct_ps)

        if ct_per_shot_all and pf_all:
            ct_arr = np.array([v for v in ct_per_shot_all if v > 0])
            pf_arr = np.array([v for v in pf_all if v > 0])
            ratio_arr = np.array([v for v in ratio_all if v > 0])
            if len(ct_arr) > 0 and len(pf_arr) > 0:
                ct_geo, ct_lo, ct_hi = _geo_stats(ct_arr)
                pf_geo, pf_lo, pf_hi = _geo_stats(pf_arr)
                ratio_geo, ratio_lo, ratio_hi = _geo_stats(ratio_arr) if len(ratio_arr) > 0 else (pf_geo / ct_geo, 0, 0)
                success_rate = len(ct_arr) / 10.0
                stars.append({
                    'nq': nq, 'ng': ng,
                    'ct_per_shot': ct_geo, 'ct_lo': ct_lo, 'ct_hi': ct_hi,
                    'pf': pf_geo, 'pf_lo': pf_lo, 'pf_hi': pf_hi,
                    'ratio': ratio_geo, 'ratio_lo': ratio_lo, 'ratio_hi': ratio_hi,
                    'label': f'Prop. $n$={nq}, $g$={ng}',
                    'hollow': success_rate < 0.8,
                })

    return stars


if __name__ == '__main__':
    main()
