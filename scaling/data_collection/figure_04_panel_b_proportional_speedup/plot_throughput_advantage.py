#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Figure 4 panel B throughput-advantage plot for proportional sampling
(bs=10, uniform batches).

X-axis: ptsbe_nshots (10, 100, 1000, 10000) — log scale
Y-axis: Throughput advantage (PTSBE / cuda-Q) — log scale
Lines:  one per config (100q_600g, 200q_1000g)

Scatter per-circuit points behind the geo-mean markers.
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

BS = 10

CONFIGS = [
    (100, 600),
    (200, 1000),
]
NSHOTS_LIST = [10, 100, 1000, 10000]

paper_colors = {
    (100, 600):   '#003F72',
    (200, 1000):  '#76B900',
}
markers = {
    (100, 600):   'o',
    (200, 1000):  's',
}
offsets = {
    (100, 600):  -0.04,
    (200, 1000):  0.04,
}


def _extract_ptsbe(filepath):
    """Return {circuit_id: shots_per_sec} from a single-nshots PTSBE file."""
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
    """Return {circuit_id: shots_per_sec} via distinct_shots / sample_time."""
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
    """Returns {(q, g, nshots): [advantage_ratios]}."""
    advantages = defaultdict(list)
    for q, g in CONFIGS:
        cf = os.path.join(DATA_DIR, f'{q}q_{g}g_100hs_cudaq.txt')
        if not os.path.isfile(cf):
            print(f'  [skip] cuda-Q file missing: {cf}')
            continue
        cudaq = _extract_cudaq(cf)
        for ns in NSHOTS_LIST:
            pf = os.path.join(DATA_DIR,
                              f'{q}q_{g}g_100hs_{BS}bs_proportional_{ns}nshots_ptsbe.txt')
            if not os.path.isfile(pf):
                continue
            ptsbe = _extract_ptsbe(pf)
            for ckt, p in ptsbe.items():
                c = cudaq.get(ckt)
                if c and c > 0 and p > 0:
                    advantages[(q, g, ns)].append(p / c)
    return advantages


def _geo_stats(arr):
    log_arr = np.log(arr)
    geo_mean = np.exp(np.mean(log_arr))
    geo_std = np.exp(np.std(log_arr))
    return geo_mean, geo_mean - geo_mean / geo_std, geo_mean * geo_std - geo_mean


def _plot(adv):
    fig, ax = plt.subplots(figsize=(10, 7))

    plotted_configs = []
    for (q, g) in CONFIGS:
        xs, means, lo, hi, raw_x, raw_y = [], [], [], [], [], []
        for ns in NSHOTS_LIST:
            ratios = adv.get((q, g, ns))
            if not ratios:
                continue
            arr = np.array(ratios)
            geo, lo_e, hi_e = _geo_stats(arr)
            xs.append(ns)
            means.append(geo)
            lo.append(lo_e)
            hi.append(hi_e)
            for r in ratios:
                raw_x.append(ns)
                raw_y.append(r)

        if not xs:
            continue

        plotted_configs.append((q, g))
        col = paper_colors[(q, g)]
        mk = markers[(q, g)]
        x_log = np.array(xs, dtype=float)
        off = offsets[(q, g)]
        x_plot = x_log * (10 ** off)

        ax.scatter([rx * (10 ** off) for rx in raw_x], raw_y,
                   color=col, alpha=0.18, s=30, marker=mk, zorder=2,
                   edgecolors='none')

        ax.errorbar(x_plot, means, yerr=[lo, hi],
                    fmt='-', color=col,
                    capsize=5, capthick=1.2,
                    linewidth=1.8, elinewidth=1.0,
                    ecolor=col + '55', zorder=3)

        ax.plot(x_plot, means, marker=mk, markersize=10,
                color=col, markeredgecolor=col, markeredgewidth=1.4,
                markerfacecolor=col, linestyle='none', zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Proportional PTSBE Shots ($m_i$)', fontsize=16)
    ax.set_ylabel('Throughput Advantage  (PTSBE / cuda-Q)', fontsize=16)
    ax.set_title('Proportional Sampling — Throughput Advantage vs nshots',
                 fontsize=16, pad=14)

    ax.set_xticks(NSHOTS_LIST)
    ax.set_xticklabels([str(ns) for ns in NSHOTS_LIST], fontsize=13)
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

    legend_handles = []
    for (q, g) in plotted_configs:
        col = paper_colors[(q, g)]
        mk = markers[(q, g)]
        legend_handles.append(Line2D(
            [0], [0], color=col, marker=mk,
            markersize=9, linewidth=1.8, linestyle='-',
            markeredgecolor=col, markeredgewidth=1.4,
            markerfacecolor=col, label=r'$n$ = ' + str(q) + r', $g$ = ' + str(g)))
    legend_handles.append(Line2D(
        [0], [0], color='gray', marker='o', markersize=7,
        linewidth=0, markeredgecolor='none',
        markerfacecolor='gray', alpha=0.3, label='Per-circuit (scatter)'))

    leg = ax.legend(handles=legend_handles, title='Config',
                    fontsize=12, title_fontsize=13,
                    framealpha=0.95, loc='best', fancybox=False,
                    edgecolor='black')
    leg.get_frame().set_linewidth(0.6)

    footnote = (
        "Markers = geometric mean, error bars = \u00b11 geometric std deviation  |  "
        "Scatter = individual circuits  |  "
        "10 random circuits per config\n"
        f"Uniform batch size = {BS}  |  nnoise_samples = 1  |  100 hypersamples"
    )
    fig.text(0.5, 0.005, footnote, ha='center', fontsize=9,
             style='italic', color='#555555', linespacing=1.5)
    fig.subplots_adjust(bottom=0.14)

    out = os.path.join(DATA_DIR, 'paper_throughput_advantage_vs_nshots.png')
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


def _plot_v2(adv):
    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(12, 8))

        plotted_configs = []
        for (q, g) in CONFIGS:
            xs, means, lo, hi, raw_x, raw_y = [], [], [], [], [], []
            for ns in NSHOTS_LIST:
                ratios = adv.get((q, g, ns))
                if not ratios:
                    continue
                arr = np.array(ratios)
                geo, lo_e, hi_e = _geo_stats(arr)
                xs.append(ns)
                means.append(geo)
                lo.append(lo_e)
                hi.append(hi_e)
                for r in ratios:
                    raw_x.append(ns)
                    raw_y.append(r)

            if not xs:
                continue

            plotted_configs.append((q, g))
            col = paper_colors[(q, g)]
            mk = markers[(q, g)]
            x_log = np.array(xs, dtype=float)
            off = offsets[(q, g)]
            x_plot = x_log * (10 ** off)

            scatter_col = _blend_hex(col, 0.28)
            ax.scatter([rx * (10 ** off) for rx in raw_x], raw_y,
                       color=scatter_col, s=30, marker=mk, zorder=2,
                       edgecolors='none')

            ebar_col = _blend_hex(col, 0.45)
            ax.errorbar(x_plot, means, yerr=[lo, hi],
                        fmt='-', color=col,
                        capsize=5, capthick=1.2,
                        linewidth=2.8, elinewidth=1.0,
                        ecolor=ebar_col, zorder=3)

            ax.plot(x_plot, means, marker=mk, markersize=11,
                    color=col, markeredgecolor=col, markeredgewidth=1.4,
                    markerfacecolor=col, linestyle='none', zorder=4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Proportional PTSBE Shots ($m_i$)', fontsize=28)
        ax.set_ylabel('Data Collection Speedup (PTSBE / cuda-Q)', fontsize=28)
        ax.set_title('Proportional Sampling \u2014 Data Collection Speedup vs nshots',
                     fontsize=28, pad=14)

        ax.set_xticks(NSHOTS_LIST)
        ax.set_xticklabels([str(ns) for ns in NSHOTS_LIST], fontsize=23)
        ax.tick_params(axis='y', labelsize=22)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        hline_col = _blend_hex('#888888', 0.6)
        for val, label in [(1e3, '1 K\u00d7'), (1e6, '1 M\u00d7')]:
            if ax.get_ylim()[0] <= val <= ax.get_ylim()[1] * 2:
                ax.axhline(y=val, color=hline_col, ls='--', lw=0.8, zorder=1)
                ax.text(1.02, val, label, transform=ax.get_yaxis_transform(),
                        ha='left', va='center', fontsize=12, color='#333333',
                        fontweight='bold', clip_on=False,
                        bbox=dict(facecolor='white', edgecolor='#999999',
                                  boxstyle='round,pad=0.25', linewidth=0.6))

        legend_handles = []
        scatter_legend_col = _blend_hex('#808080', 0.4)
        for (q, g) in plotted_configs:
            col = paper_colors[(q, g)]
            mk = markers[(q, g)]
            legend_handles.append(Line2D(
                [0], [0], color=col, marker=mk,
                markersize=12, linewidth=2.8, linestyle='-',
                markeredgecolor=col, markeredgewidth=1.4,
                markerfacecolor=col, label=r'$n$ = ' + str(q) + r', $g$ = ' + str(g)))
        legend_handles.append(Line2D(
            [0], [0], color=scatter_legend_col, marker='o', markersize=9,
            linewidth=0, markeredgecolor='none',
            markerfacecolor=scatter_legend_col, label='Per-circuit (scatter)'))

        leg = ax.legend(handles=legend_handles, title='Config',
                        fontsize=23, title_fontsize=24,
                        framealpha=1.0, loc='best', fancybox=False,
                        edgecolor='black')
        leg.get_frame().set_linewidth(0.6)

        fig.tight_layout(pad=1.5)

        out_pdf = os.path.join(DATA_DIR, 'paper_throughput_advantage_vs_nshots_v2.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(DATA_DIR, 'paper_throughput_advantage_vs_nshots_v2.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_png}')

        plt.close(fig)


def _plot_v3(adv):
    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(12, 8))

        plotted_configs = []
        for (q, g) in CONFIGS:
            xs, means, lo, hi, raw_x, raw_y = [], [], [], [], [], []
            for ns in NSHOTS_LIST:
                ratios = adv.get((q, g, ns))
                if not ratios:
                    continue
                arr = np.array(ratios)
                geo, lo_e, hi_e = _geo_stats(arr)
                xs.append(ns)
                means.append(geo)
                lo.append(lo_e)
                hi.append(hi_e)
                for r in ratios:
                    raw_x.append(ns)
                    raw_y.append(r)

            if not xs:
                continue

            plotted_configs.append((q, g))
            col = paper_colors[(q, g)]
            mk = markers[(q, g)]
            x_log = np.array(xs, dtype=float)
            off = offsets[(q, g)]
            x_plot = x_log * (10 ** off)

            scatter_col = _blend_hex(col, 0.28)
            ax.scatter([rx * (10 ** off) for rx in raw_x], raw_y,
                       color=scatter_col, s=30, marker=mk, zorder=2,
                       edgecolors='none')

            ebar_col = _blend_hex(col, 0.45)
            ax.errorbar(x_plot, means, yerr=[lo, hi],
                        fmt='-', color=col,
                        capsize=5, capthick=1.2,
                        linewidth=2.8, elinewidth=1.0,
                        ecolor=ebar_col, zorder=3)

            ax.plot(x_plot, means, marker=mk, markersize=11,
                    color=col, markeredgecolor=col, markeredgewidth=1.4,
                    markerfacecolor=col, linestyle='none', zorder=4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Proportional PTSBE Shots ($m_i$)', fontsize=28)
        ax.set_ylabel('Data Collection Speedup (PTSBE / cuda-Q)', fontsize=28)

        ax.set_xticks(NSHOTS_LIST)
        ax.set_xticklabels([str(ns) for ns in NSHOTS_LIST], fontsize=23)
        ax.tick_params(axis='y', labelsize=22)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        hline_col = _blend_hex('#888888', 0.6)
        for val, label in [(1e3, '1 K\u00d7'), (1e6, '1 M\u00d7')]:
            if ax.get_ylim()[0] <= val <= ax.get_ylim()[1] * 2:
                ax.axhline(y=val, color=hline_col, ls='--', lw=0.8, zorder=1)
                ax.text(1.02, val, label, transform=ax.get_yaxis_transform(),
                        ha='left', va='center', fontsize=12, color='#333333',
                        fontweight='bold', clip_on=False,
                        bbox=dict(facecolor='white', edgecolor='#999999',
                                  boxstyle='round,pad=0.25', linewidth=0.6))

        legend_handles = []
        scatter_legend_col = _blend_hex('#808080', 0.4)
        for (q, g) in plotted_configs:
            col = paper_colors[(q, g)]
            mk = markers[(q, g)]
            legend_handles.append(Line2D(
                [0], [0], color=col, marker=mk,
                markersize=12, linewidth=2.8, linestyle='-',
                markeredgecolor=col, markeredgewidth=1.4,
                markerfacecolor=col, label=r'$n$ = ' + str(q) + r', $g$ = ' + str(g)))
        legend_handles.append(Line2D(
            [0], [0], color=scatter_legend_col, marker='o', markersize=9,
            linewidth=0, markeredgecolor='none',
            markerfacecolor=scatter_legend_col, label='Per-circuit (scatter)'))

        leg = ax.legend(handles=legend_handles, title='Config',
                        fontsize=23, title_fontsize=24,
                        framealpha=1.0, loc='best', fancybox=False,
                        edgecolor='black')
        leg.get_frame().set_linewidth(0.6)

        fig.tight_layout(pad=1.5)

        out_pdf = os.path.join(DATA_DIR, 'paper_throughput_advantage_vs_nshots_v3.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(DATA_DIR, 'paper_throughput_advantage_vs_nshots_v3.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_png}')

        plt.close(fig)


def main():
    adv = _collect_advantages()

    print(f'Proportional3 (bs={BS}) — Throughput advantage summary:')
    print(f'{"Config":<22} {"nshots":>6}  {"n":>3}  {"GeoMean":>12}  {"Min":>12}  {"Max":>12}')
    print('-' * 70)
    for (q, g, ns), ratios in sorted(adv.items()):
        if not ratios:
            continue
        a = np.array(ratios)
        geo = np.exp(np.mean(np.log(a)))
        print(f'{q}q_{g}g {ns:>6}  {len(a):>3}  {geo:>12,.1f}\u00d7  '
              f'{np.min(a):>12,.1f}\u00d7  {np.max(a):>12,.1f}\u00d7')

    _plot_v4(adv)


def _plot_v4(adv):
    """V4: larger fonts across all elements."""
    paper_colors = {
        (100, 600):   '#003F72',
        (200, 1000):  '#76B900',
    }
    markers = {
        (100, 600):   'o',
        (200, 1000):  's',
    }
    offsets = {
        (100, 600):  -0.04,
        (200, 1000):  0.04,
    }

    with mpl.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(13, 9))

        plotted_configs = []
        for (q, g) in CONFIGS:
            xs, means, lo, hi, raw_x, raw_y = [], [], [], [], [], []
            for ns in NSHOTS_LIST:
                ratios = adv.get((q, g, ns))
                if not ratios:
                    continue
                arr = np.array(ratios)
                geo, lo_e, hi_e = _geo_stats(arr)
                xs.append(ns)
                means.append(geo)
                lo.append(lo_e)
                hi.append(hi_e)
                for r in ratios:
                    raw_x.append(ns)
                    raw_y.append(r)

            if not xs:
                continue

            plotted_configs.append((q, g))
            col = paper_colors[(q, g)]
            mk = markers[(q, g)]
            x_log = np.array(xs, dtype=float)
            off = offsets[(q, g)]
            x_plot = x_log * (10 ** off)

            scatter_col = _blend_hex(col, 0.28)
            ax.scatter([rx * (10 ** off) for rx in raw_x], raw_y,
                       color=scatter_col, s=30, marker=mk, zorder=2,
                       edgecolors='none')

            ebar_col = _blend_hex(col, 0.45)
            ax.errorbar(x_plot, means, yerr=[lo, hi],
                        fmt='-', color=col,
                        capsize=5, capthick=2.0,
                        linewidth=3.6, elinewidth=3.6,
                        ecolor=ebar_col, zorder=3)

            ax.plot(x_plot, means, marker=mk, markersize=15,
                    color=col, markeredgecolor=col, markeredgewidth=1.4,
                    markerfacecolor=col, linestyle='none', zorder=4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Proportional PTSBE Shots ($m_i$)', fontsize=32)
        ax.set_ylabel('Data Collection Speedup (PTSBE / cuda-Q)', fontsize=32)

        ax.set_xticks(NSHOTS_LIST)
        ax.set_xticklabels([str(ns) for ns in NSHOTS_LIST], fontsize=27)
        ax.tick_params(axis='y', labelsize=28)
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        grid_col = _blend_hex('#000000', 0.15)
        ax.grid(True, which='major', ls='--', color=grid_col)

        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

        legend_handles = []
        scatter_legend_col = _blend_hex('#808080', 0.4)
        for (q, g) in plotted_configs:
            col = paper_colors[(q, g)]
            mk = markers[(q, g)]
            legend_handles.append(Line2D(
                [0], [0], color=col, marker=mk,
                markersize=15, linewidth=3.6, linestyle='-',
                markeredgecolor=col, markeredgewidth=1.4,
                markerfacecolor=col, label=r'$n$ = ' + str(q) + r', $g$ = ' + str(g)))
        legend_handles.append(Line2D(
            [0], [0], color=scatter_legend_col, marker='o', markersize=11,
            linewidth=0, markeredgecolor='none',
            markerfacecolor=scatter_legend_col, label='Per-circuit (scatter)'))

        leg = ax.legend(handles=legend_handles, title='Config',
                        fontsize=26, title_fontsize=27,
                        framealpha=1.0, loc='best', fancybox=False,
                        edgecolor='black')
        leg.get_frame().set_linewidth(0.6)

        fig.tight_layout(pad=1.5)

        out_pdf = os.path.join(DATA_DIR, 'paper_throughput_advantage_vs_nshots_v4.pdf')
        fig.savefig(out_pdf, format='pdf', bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_pdf}')

        out_png = os.path.join(DATA_DIR, 'paper_throughput_advantage_vs_nshots_v4.png')
        fig.savefig(out_png, dpi=600, bbox_inches='tight', pad_inches=0.3)
        print(f'Saved  {out_png}')

        plt.close(fig)


if __name__ == '__main__':
    main()
