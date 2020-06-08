#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from mpl_tools.helpers import add_to_labeled_items, add_colorbar, savefig
from matplotlib.colors import LinearSegmentedColormap
import pickle
import itertools as it

colors = [(0.4, 0.4, 0.4), (0.8, 0.8, 0.8), (0.4, 0.8, 0.4)]  #
cm = LinearSegmentedColormap.from_list('cmsimple', colors, N=3)

def MakeEqualScale(edges):
    """Convert any set of ticks to evenly spaced ticks"""
    widths = edges[1:] - edges[:-1]
    def forward(values):
        idxs = np.searchsorted(edges, values, side='right')
        ret = np.zeros_like(values)

        idxs-=1
        idxs[idxs<0]=0
        idxs[idxs>=widths.size]=widths.size-1
        ret = idxs + (values - edges[idxs])/widths[idxs]

        return ret

    def inverse(values):
        idxs = np.array(values, dtype='i')
        idxs[idxs<0]=0
        idxs[idxs>=widths.size]=widths.size-1

        return edges[idxs] + widths[idxs]*(values - idxs)

    return forward, inverse

def show_values(pc, fmt="%.2f", **kw):
    pc.update_scalarmappable()
    ax = plt.gca()
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:3:2].mean(0)
        if np.mean(color[:3]) > 0.5:
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)

def plot_boxes(low, high, data=None, title=None, scale=False):
    fig = plt.figure(figsize=(8, 6))
    if title is None:
        title = 'Energy limits map'
    else:
        title = 'Energy limits map: '+title

    ax = plt.subplot(111, xlabel='E low', ylabel='E high', title=title)
    # ax.minorticks_on()

    if scale:
        fwd_x, inv_x = MakeEqualScale(low)
        ax.set_xscale('function', functions=(fwd_x, inv_x))
        fwd_y, inv_y = MakeEqualScale(high)
        ax.set_yscale('function', functions=(fwd_y, inv_y))
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

    if data is None:
        # Helper rectangle
        rpos = (0.60, 0.05)
        rwidth, rheight = 0.1, 0.1
        hlen = rwidth*0.1
        hwidth = hlen
        rcenter_x = rpos[0]+rwidth*0.5
        rcenter_y = rpos[1]+rheight*0.5
        rect_example = Rectangle(rpos, rwidth, rheight, color='white', zorder=zorder, transform=ax.transAxes)
        ax.add_artist(rect_example)
        # Arrows
        opts = dict(zorder=zorder+2, head_width=hwidth, head_length=hlen, ec='black', fc='black', transform=ax.transAxes)
        ax.arrow(rpos[0], rcenter_y, rwidth*0.4-hlen, 0.0, **opts)
        ax.arrow(rcenter_x, rcenter_y+rheight*0.1, 0.0, rheight*0.4-hlen, **opts)
        # # # Highlight sides
        opts = dict(linewidth=2.5, color='black', zorder=zorder+1, transform=ax.transAxes, head_width=0.0, head_length=0.0)
        ax.arrow(rpos[0], rpos[1], 0.0, rheight, **opts)
        ax.arrow(rpos[0], rpos[1]+rheight, rwidth, 0.0, **opts)

    if data is None:
        # Total
        rect_total = Rectangle((eminimal, 10.0), 0.3, 2.0, color='magenta', zorder=zorder)
        ax.add_artist(rect_total)

    # Minimal
    if data is None:
        rect_min = Rectangle((emin, emax-ew), ew, ew, color='green', zorder=zorder)
        ax.add_artist(rect_min)
    else:
        # rect_min = Rectangle((emin, emax-ew), ew, ew, fc='none', ec='yellow', linestyle='dashed', zorder=zorder)
        pass
    goodlineopts = dict(zorder=zorder+1, linestyle='--', color='yellow', alpha=0.5)
    goodline = ax.vlines(emin, emax, emaximal, **goodlineopts)
    ax.hlines(emax, eminimal, emin, **goodlineopts)

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
    Data = np.ma.array(np.zeros_like(L, dtype='d'), mask=np.zeros_like(L, dtype='i'))
    for emin, emax, fun, success in data:
        imin = np.searchsorted(low, emin)
        imax = np.searchsorted(high, emax)-1
        Data[imin, imax] = fun
        Data.mask[imin, imax] = not success

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

    if scale:
        show_values(c, fontsize='x-small')

    ax.grid(**gridopts)

    import IPython; IPython.embed()

def plot_combination(split, title):
    #
    # Combination
    #
    fig = plt.figure()
    ax = plt.subplot(111, xlabel='E split, MeV', ylabel=r'$\Delta \chi^2$', title='Split test: '+title)
    ax.minorticks_on()
    ax.grid()

    left_x, left_y = split['left']
    right_x, right_y = split['right']

    left_x = np.around(left_x, 6)
    ax.plot(left_x, left_y, label='left: [0.7, x] MeV')
    ax.plot(right_x, right_y, label='right: [x, 12] MeV')

    idx_right = np.in1d(right_x, left_x)
    idx_left  = np.in1d(left_x, right_x)

    both_x = left_x[idx_left]
    both_y = left_y[idx_left] + right_y[idx_right]
    ax.plot(both_x, both_y, label='combined: uncorrelated sum')
    ax.legend()


def load_data(args):
    data = {}

    threshold, ceiling = 0.7, 12.0
    for i, inp in enumerate(args.input):
        dataset = []
        emin_all = set()
        emax_all = set()
        split_left = ()
        split_right = ()
        for name in inp:
            with open(name, 'rb') as f:
                d=pickle.load(f, encoding='latin1')['fitresult']['min']
                emin, emax = d['info']['emin'], d['info']['emax']
                fun = d['fun']
                emin_all.add(emin)
                emax_all.add(emax)
                dataset.append((emin, emax, fun, d['success']))

                if np.isclose(emin, threshold):
                    split_left += (emax, fun),
                if np.isclose(emax, ceiling):
                    split_right += (emin, fun),

        emin_all = list(sorted(emin_all))
        emax_all = list(sorted(emax_all))
        emin_all.append(emax_all[-1])
        emax_all = [emin_all[0]] + emax_all
        emin_all, emax_all = np.array(emin_all), np.array(emax_all)

        idata = data[str(i)] = {}
        idata['scan'] = (emin_all, emax_all, dataset)

        split = idata['split'] = {}
        split['left'] = np.array(split_left).T
        split['right'] = np.array(split_right).T

    return data

def main(args):
    low = np.concatenate( ( [0.7], np.arange(1.0, 8.0, 0.5) ) )
    high = np.concatenate( (np.arange(1.5, 6.0, 0.5), [9.0, 12.0] ) )
    plot_boxes(low, high)
    savefig(args.output, suffix='_map')
    hasmap=True

    plot_boxes(low, high, scale=True)
    savefig(args.output, suffix='_map_scaled')

    data = load_data(args)
    for i, (idata, title) in enumerate(it.zip_longest(data.values(), args.title)):
        low, high, scan = idata['scan']
        plot_boxes(low, high, scan, title=title)
        savefig(args.output, suffix='_{}_full'.format(i))

        ax=plt.gca()
        ax.set_xlim(right=4.0)
        savefig(args.output, suffix='_{}_zoom'.format(i))

        plot_boxes(low, high, scan, title=title, scale=True)
        savefig(args.output, suffix='_{}_scaled_full'.format(i))

        ax=plt.gca()
        ax.set_xlim(right=4.0)
        savefig(args.output, suffix='_{}_scaled_zoom'.format(i))

        plot_combination(idata['split'], title)
        savefig(args.output, suffix='_{}_split'.format(i))

    plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', nargs='+', action='append', help='input files')
    parser.add_argument('--title', default=[], action='append', help='titles')
    parser.add_argument('-o', '--output', help='output file')

    args=parser.parse_args()
    main(args)
