#!/usr/bin/env python

from __future__ import print_function
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from mpl_tools.helpers import add_to_labeled_items, add_colorbar, savefig
from matplotlib.colors import LinearSegmentedColormap
import pickle
import itertools as it

colors = [(0.4, 0.4, 0.4), (0.8, 0.8, 0.8), (0.4, 0.8, 0.4)]  #
cm = LinearSegmentedColormap.from_list('cmsimple', colors, N=3)

def show_values(pc, fmt="%.2f", **kw):
    from itertools import izip
    pc.update_scalarmappable()
    ax = plt.gca()
    for p, color, value in izip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:3:2].mean(0)
        if np.mean(color[:3]) > 0.5:
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def plot_boxes(low, high, data=None, title=None):
    fig = plt.figure(figsize=(8, 6))
    if title is None:
        title = 'Energy limits map'
    else:
        title = 'Energy limits map: '+title

    ax = plt.subplot(111, xlabel='E low', ylabel='E high', title=title)
    # ax.minorticks_on()
    ax.set_xticks(low)
    ax.set_yticks(high)

    eminimal = low[0]
    emaximal = high[-1]

    L, H = np.meshgrid(low, high, indexing='ij')
    emin, emax, ew = 1.5, 4.0, 0.5
    gridopts=dict(color='red', alpha=0.4, linestyle='dashed')

    #
    # Plot stuff
    #
    zorder=10

    # Helper rectangle
    rpos = (4.0,2.0)
    rwidth, rheight = 1.0, 1.0
    rcenter_x = rpos[0]+rwidth*0.5
    rcenter_y = rpos[1]+rheight*0.5
    rect_example = Rectangle(rpos, rwidth, rheight, color='white', zorder=zorder)
    ax.add_artist(rect_example)
    # Arrows
    hlen = 0.1
    opts = dict(zorder=zorder+2, head_width=0.1, head_length=hlen, ec='black', fc='black')
    ax.arrow(rpos[0], rcenter_y, rwidth*0.4-hlen, 0.0, **opts)
    ax.arrow(rcenter_x, rcenter_y+rheight*0.1, 0.0, rheight*0.4-hlen, **opts)
    # Highlight sides
    ax.vlines(rpos[0], rpos[1], rpos[1]+rheight, color='gray', zorder=zorder+1, linewidth=2.0)
    ax.hlines(rpos[1]+rheight, rpos[0], rpos[0]+rwidth,  color='gray', zorder=zorder+1, linewidth=2.0)

    if data is None:
        # Total
        rect_total = Rectangle((eminimal, 10.0), 0.3, 2.0, color='magenta', zorder=zorder)
        ax.add_artist(rect_total)

    # Minimal
    if data is None:
        rect_min = Rectangle((emin, emax-ew), ew, ew, color='green', zorder=zorder)
    else:
        rect_min = Rectangle((emin, emax-ew), ew, ew, fc='none', ec='yellow', linestyle='dashed', zorder=zorder)
    ax.add_artist(rect_min)
    goodlineopts = dict(zorder=zorder+1, linestyle='--', color='yellow')
    goodline = ax.vlines(emin+ew, emax-ew, emaximal, **goodlineopts)
    ax.hlines(emax-ew, eminimal, emin+ew, **goodlineopts)

    if data is None:
        #
        # Legend
        #
        rect_forbidden = Rectangle((emin, emax-ew), ew, ew, color=cm(0))
        rect_bad = Rectangle((emin, emax-ew), ew, ew, color=cm(1))
        rect_acceptable = Rectangle((emin, emax-ew), ew, ew, color=cm(2))
        handles = [rect_example, rect_total, rect_min, goodline, rect_acceptable, rect_forbidden, rect_bad]
        labels =  ['Example', 'Total: {:.1f}$-${:.0f}'.format(eminimal, 12.0), 'Minimal: {:.1f}$-${:.0f}'.format(emin, emax), 'Acceptable', 'Acceptable', 'Forbidden', 'Bad']
        ax.legend(handles, labels, loc='lower right')

        #
        # Example plot
        #
        data = np.ones_like(L, dtype='i')*2.0
        data[L>emin] = 1
        data[H<emax-ew] = 1
        data[L>H] = 0
        data=data[:-1,:-1]
        ax.pcolormesh(L, H, data, vmin=0.1, cmap=cm)

        ax.grid(**gridopts)

        return

    #
    # Data
    #
    Data = np.zeros_like(L, dtype='d')
    for emin, emax, fun in data:
        imin = np.searchsorted(low, emin)
        imax = np.searchsorted(high, emax)-1
        Data[imin, imax] = fun

        if fun>12.5:
            # Data[imin, imax] = -1
            print('Strange value below:')
        print(
                '{index1:02d} {emin} in ({emin1}, {emin2})'
                '\t'
                '{index2:02d} {emax} in ({emax1}, {emax2})'
                '\t'
                '{fun}'.format(
                    index1=imin, emin=emin, emin1=low[imin] if len(low)>imin else -1, emin2=low[imin+1] if len(low)-1>imin else -1,
                    index2=imax, emax=emax, emax1=high[imax] if len(high)>imax else -1, emax2=high[imax+1] if len(high)-1>imax else -1,
                    fun=fun
                    )
                )
    c = ax.pcolormesh(L, H, Data, vmin=0.1)
    add_colorbar(c)
    show_values(c)
    ax.grid(**gridopts)

def load_data(args):
    datasets = []
    for inp in args.input:
        data = []
        emin_all = set()
        emax_all = set()
        for name in inp:
            with open(name, 'r') as f:
                d=pickle.load(f)
                emin, emax = float(d['emin']), float(d['emax'])
                fun = d['fun']
                emin_all.add(emin)
                emax_all.add(emax)
                data.append((emin, emax, fun))

        emin_all = list(sorted(emin_all))
        emax_all = list(sorted(emax_all))
        emin_all.append(emax_all[-1])
        emax_all = [emin_all[0]] + emax_all
        datasets.append((emin_all, emax_all, data))

    return datasets

def main(args):
    hasmap = False

    datasets = load_data(args)

    for i, ((low, high, data), title) in enumerate(it.izip_longest(datasets, args.title)):
        if not hasmap:
            plot_boxes(low, high)
            savefig(args.output, suffix='_map')
            hasmap=True

        plot_boxes(low, high, data, title=title)
        savefig(args.output, suffix='_{}_full'.format(i))

        ax=plt.gca()
        ax.set_xlim(right=4.0)
        savefig(args.output, suffix='_{}_zoom'.format(i))

    plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', nargs='+', action='append', help='input files')
    parser.add_argument('--title', default=[], action='append', help='titles')
    parser.add_argument('--output', help='output file')

    args=parser.parse_args()
    main(args)
