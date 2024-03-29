#!/usr/bin/env python3

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from mpl_tools.helpers import add_to_labeled_items, add_colorbar, savefig
from matplotlib.colors import LinearSegmentedColormap
import pickle
import itertools as it

dchi2 = r'$\Delta \chi^2$'
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
        x, y = p.vertices[2:4].mean(0)
        x-=0.1*(p.vertices[2,0]-p.vertices[1,0])
        if np.mean(color[:3]) > 0.5:
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="right", va="center", color=color, **kw)

def plot_boxes(dataall=None, title=None, scale=False, low=None, high=None, output=None, suffix=(), *, args=None):
    if dataall is None:
        assert low is not None
        assert high is not None
        data = None
    else:
        low, high, data = dataall['scan']
        split = dataall['split']

    xlabel='E low, MeV'
    ylabel='E high, MeV'
    eminimal = low[0]
    emaximal = high[-1]
    if scale:
        fwd_x, inv_x = MakeEqualScale(low)
        fwd_y, inv_y = MakeEqualScale(high)


    # Prepare data
    L, H = np.meshgrid(low, high, indexing='ij')
    W = np.around(H[1:, 1:] - L[:-1, :-1], 6)
    Center = np.around(H[1:, 1:] + L[:-1, :-1], 6)*0.5
    emin, emax, ew = 1.5, 4.0, 0.5
    gridopts=dict(color='red', alpha=0.4, linestyle='dashed')

    #
    # Make figure
    #
    fig = plt.figure(figsize=(8, 6))
    title_suffix = ''
    if title:
        title_suffix = ': '+title
    ax = plt.subplot(111, xlabel=xlabel, ylabel=ylabel, title='Energy limits map'+title_suffix)
    if scale:
        ax.set_xscale('function', functions=(fwd_x, inv_x))
        ax.set_yscale('function', functions=(fwd_y, inv_y))
    ax.xaxis.set_tick_params(top=True, labeltop=True, which='both')
    ax.set_xticks(low)
    ax.set_yticks(high)
    plt.subplots_adjust(bottom=0.11)
    if args.equal:
        ax.set_aspect('equal')

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
        goodlineopts = dict(zorder=zorder+1, linestyle='--', color='yellow', alpha=0.5)
        goodline = ax.vlines(emin, emax, emaximal, **goodlineopts)
        ax.hlines(emax, eminimal, emin, **goodlineopts)
    else:
        # rect_min = Rectangle((emin, emax-ew), ew, ew, fc='none', ec='yellow', linestyle='dashed', zorder=zorder)
        pass

    if data is None:
        #
        # Legend
        #
        rect_forbidden = Rectangle((emin, emax-ew), ew, ew, color=cm(0))
        rect_bad = Rectangle((emin, emax-ew), ew, ew, color=cm(1))
        rect_acceptable = Rectangle((emin, emax-ew), ew, ew, color=cm(2))
        handles = [rect_example, rect_total, rect_min, rect_acceptable, rect_forbidden, rect_bad]
        labels =  ['Example', 'Total: {:.1f}$-${:.0f}'.format(eminimal, 12.0), 'Minimal: {:.1f}$-${:.0f}'.format(emin, emax), 'Acceptable', 'Acceptable', 'Forbidden', 'Bad']
        ax.legend(handles, labels, loc='center right')

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
    Data = np.ma.array(np.zeros_like(L, dtype='d'), mask=np.zeros_like(L, dtype='i'))[:-1,:-1]
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

    dchi2loc = dchi2
    data_scale_factor=1.0
    vmin=0.1
    if args.relative:
        dchi2loc+=', %'
        data_scale_factor=100.0/Data.max()
        Data*=data_scale_factor
        vmin=0.0
    c = ax.pcolormesh(L, H, Data, vmin=vmin)
    add_colorbar(c, label=dchi2loc)

    if scale:
        show_values(c, fontsize='x-small')

    ax.grid(**gridopts)

    lines2opts = dict(color='red', linewidth=1, linestyle='-')
    ax.axvline(low[1], **lines2opts)
    ax.axhline(high[-2], **lines2opts)
    axd = ax

    ax.set_xlim(right=3.4)
    savefig(output, suffix=suffix+('zoom',))

    if not scale:
        return axd, None, None
    #
    # Test data
    #

    # fig = plt.figure()
    # ax = plt.subplot(111, xlabel=xlabel, ylabel=ylabel, title='Range, MeV')
    # ax.minorticks_on()
    # ax.grid()

    # ax.set_xscale('function', functions=(fwd_x, inv_x))
    # ax.set_yscale('function', functions=(fwd_y, inv_y))
    # ax.set_xticks(low)
    # ax.set_yticks(high)

    # c = ax.pcolormesh(L, H, W)
    # add_colorbar(c)
    # show_values(c, fontsize='x-small')

    #
    # Combination
    #
    fig = plt.figure()
    axc = plt.subplot(111, xlabel='E split, MeV', ylabel=dchi2loc, title='Split test'+title_suffix)
    axc.xaxis.set_tick_params(top=True, labeltop=True, which='both')
    axc.minorticks_on()
    axc.grid()

    left_x, left_y = split['left']
    right_x, right_y = split['right']
    left_y*=data_scale_factor
    right_y*=data_scale_factor

    left_x = np.around(left_x, 6)
    axc.plot(left_x, left_y, label='left: [0.7, x] MeV')
    axc.plot(right_x, right_y, label='right: [x, 12] MeV')

    idx_right = np.in1d(right_x, left_x)
    idx_left  = np.in1d(left_x, right_x)

    both_x = left_x[idx_left]
    both_y = left_y[idx_left] + right_y[idx_right]
    axc.plot(both_x, both_y, label='combined: uncorrelated sum')

    idx = np.argmin(both_y)
    worst_split_low_idx = np.argwhere(low==both_x[idx])[0,0]
    worst_split_high_idx = np.argwhere(high==both_x[idx])[0,0]-1

    worst_dchi2 = both_x[idx]
    worst_split = both_y[idx]
    if args.relative:
        label = 'worst split: {}={:.2f}% at {} MeV'.format(dchi2, worst_split, worst_dchi2)
    else:
        label = 'worst split: {}={:.2f} at {} MeV'.format(dchi2, worst_split, worst_dchi2)
    axc.axvline(both_x[idx], linestyle='--', alpha=0.5, label=label)
    axc.legend()

    savefig(output, suffix=suffix+('split',))

    #
    # Moving window
    #
    fig = plt.figure()
    ax = plt.subplot(111, xlabel='Window center, MeV', ylabel=dchi2loc, title='Moving window'+title_suffix)
    ax.xaxis.set_tick_params(top=True, labeltop=True, which='both')
    ax.minorticks_on()
    ax.grid()
    plt.subplots_adjust(right=0.81)

    Wunique = np.unique(W)

    maxpath_x = []
    maxpath_y = []
    maxpath_fun = []
    counter = 0
    for w in Wunique:
        if w<=0.0:
            continue

        mask = W==w
        x = Center[mask]
        y = Data[mask]

        imax = np.argmax(y)
        xmax, ymax = x[imax], y[imax]
        if mask.sum()>=3:
            maxpath_fun.append(ymax)

        if mask.sum()>=3:
            color = ax.plot(x, y, label=str(w))[0].get_color()

            # if mask.sum()>=4:
                # eopts=dict(alpha=0.4, linewidth=3)
            # else:
            eopts=dict(alpha=1, linewidth=0.5, capsize=3)
            ax.errorbar([xmax], [ymax], xerr=w*0.5, fmt='o-', markerfacecolor='none', color=color, **eopts)
            counter+=1

            if counter%5==0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1], title='Width:', bbox_to_anchor=(1.0, 1.15), loc='upper left', labelspacing=0.1)
                savefig(output, suffix=suffix+('window',str(counter)))

            if counter==8:
                ax.axhline(worst_dchi2, linestyle='--', label='Min split', alpha=0.6, color='blue')

    ax.axhline(np.max(Data), linestyle='--', label='Full')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Width:', bbox_to_anchor=(1.0, 1.15), loc='upper left', labelspacing=0.1)
    savefig(output, suffix=suffix+('window',))

    #
    # Max path on the main plot
    #
    plt.sca(axd)

    maxpath_fun = sorted(maxpath_fun)
    fun_prev = None
    while True:
        fun = maxpath_fun[-1]
        mask = np.isclose(Data, fun)
        idxs = np.argwhere(mask)

        cmpto=[]
        for idx in idxs:
            if idx[0]>0:
                app = Data[idx[0]-1, idx[1]]
                if app!=fun and app!=fun_prev:
                    cmpto.append(app)
            if idx[1]<Data.shape[1]-1:
                app = Data[idx[0], idx[1]+1]
                if app!=fun and app!=fun_prev:
                    cmpto.append(app)

        if not cmpto:
            break
        maxpath_fun.append(max(cmpto))
        fun_prev = fun

    Lcenter = 0.15*L[1:,1:] +0.85*L[:-1,1:]
    Hcenter = 0.20*H[:-1,1:]+0.80*H[:-1,:-1]
    for fun in maxpath_fun:
        mask = np.isclose(Data, fun)
        maxpath_x.append(Lcenter[mask][0])
        maxpath_y.append(Hcenter[mask][0])

    # Max path
    xworst = [Lcenter[0, 0],                    Lcenter[worst_split_low_idx, -1]]
    yworst = [Hcenter[0, worst_split_high_idx], Hcenter[0, -1]]

    equiv_idx = np.argmin((Data-worst_dchi2)**2)
    equiv_idx = np.unravel_index(equiv_idx, Data.shape)
    equiv_idx1 = np.argmin((maxpath_fun-worst_dchi2)**2)
    x = [Lcenter[equiv_idx[0], equiv_idx[1]], maxpath_x[equiv_idx1]]
    y = [Hcenter[equiv_idx[0], equiv_idx[1]], maxpath_y[equiv_idx1]]

    # plot
    axd.plot(xworst, yworst, 'o', color='cyan', markerfacecolor='none', label='Worst split')
    axd.legend(bbox_to_anchor=(1.2, -0.16), loc='lower right', ncol=3, fontsize='small', numpoints=2)

    savefig(output, suffix=suffix+('zoom_2',))

    axd.plot(maxpath_x, maxpath_y, 'o', color='red', markerfacecolor='none', label='Moving window maximum position')
    axd.plot(x, y, 'o', color='red', alpha=0.8, markerfacecolor='red', label='Closest to worst split')
    axd.legend(bbox_to_anchor=(1.2, -0.16), loc='lower right', ncol=3, fontsize='small', numpoints=2)

    savefig(output, suffix=suffix+('zoom_3',))

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
                if np.isclose(emax, 9.0):
                    continue
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
    if args.plot_map:
        low = np.concatenate( ( [0.7], np.arange(1.0, 8.0, 0.5) ) )
        high = np.concatenate( (np.arange(1.5, 6.0, 0.5), [9.0, 12.0] ) )
        plot_boxes(low=low, high=high, args=args)
        savefig(args.output, suffix='_map')
        plt.close()

        plot_boxes(low=low, high=high, scale=True, args=args)
        savefig(args.output, suffix='_map_scaled')
        plt.close()

    data = load_data(args)
    for i, (idata, title) in enumerate(it.zip_longest(data.values(), args.title)):
        plot_boxes(idata, title=title, scale=True, output=args.output, suffix=('_{}_scaled'.format(i),), args=args)

    plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', nargs='+', action='append', help='input files')
    parser.add_argument('--title', default=[], action='append', help='titles')
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('--plot-map', '--map', action='store_true', help='plot map')
    parser.add_argument('-r', '--relative', action='store_true', help='plot relative map')
    parser.add_argument('--equal', '--eq', action='store_true', help='aspect ratio: eq')

    args=parser.parse_args()
    main(args)
