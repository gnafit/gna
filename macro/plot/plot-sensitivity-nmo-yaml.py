#!/usr/bin/env python
# encoding: utf-8

"""Read a set of yaml files with fit results and plot sensitivity to neutrino mass hierachy (NMO)

It expects the fit data: the model (IO) fit against asimov prediction of model with NO hypothesis

"""

from __future__ import print_function
from matplotlib import pyplot as plt
from yaml import load, Loader
from mpl_tools.helpers import savefig
import numpy as np

import itertools as I
import inspect
def bars_relative(labels, values, y=None, height=0.9, **kwargs):
    if len(labels)!=len(values):
        raise Exception('Labels and values length should have similar length')

    ax=plt.gca()

    label_style_fcn = kwargs.pop('label_style_fcn', lambda col, i, value, shift, text: None)

    facecolors = kwargs.pop('facecolors', None)
    if isinstance(facecolors, list) and len(facecolors)>0:
        if isinstance(facecolors[0], tuple):
            def get_color(i):
                try:
                    ecolor = facecolors[i]
                    return ecolor[0] if changes[i]<0 else ecolor[1]
                except:
                    return None
        else:
            def get_color(i):
                try:
                    return facecolors[i]
                except:
                    return None
    elif isinstance(facecolors, tuple):
        def get_color(i):
            try:
                return facecolors[0] if changes[i]<0 else facecolors[1]
            except:
                return None
    elif isinstance(facecolors, str):
        def get_color(i):
            return facecolors
    elif inspect.isroutine(facecolors):
        get_color=facecolors
    else:
        def get_color(i):
            return None

    features = kwargs.pop('features', None)
    if isinstance(features, dict):
        def get_feature(i):
            return features.get(i, {})
    elif isinstance(features, list):
        def get_feature(i):
            try:
                ret = features[i]
                if not isinstance(ret, dict):
                    return {}
                return {}
            except:
                return {}
    elif inspect.isroutine(features):
        get_feature = features
    else:
        def get_feature(i):
            return {}

    if y is None:
        y=np.arange(len(labels))
    else:
        if len(y)!=len(values):
            raise Exception('y and values length should have similar length')

    if isinstance(height, (int, float)):
        height_it = I.repeat(height)
    else:
        height_it = height

    align = kwargs.pop('align', 'center')
    if not align in ('top', 'bottom', 'center'):
        raise Exception('Invalid alignment: '+align)

    ticks_at = y
    if align=='top':
        y = y-height
    elif align=='center':
        y = y-height*0.5

    prev_value = kwargs.pop('initial_value', 0.0)
    values = np.asanyarray(values)
    changes = values.copy()
    changes[1:]-=values[:-1]
    changes[0]=values[0]-prev_value
    for i, (label, value, change, yi, h) in enumerate(zip(labels, values, changes, y, height_it)):
        # label, value
        # chi2_prev = self.chi2_prev[i]
        # shift = self.shift[i]
        # ytop, ybottom, ytop_prev = self.ytop[i], self.ybottom[i], self.ytop_prev[i]
        # ax.broken_barh([(chi2_prev, shift)], (ytop, ybottom-ytop), facecolor=self.facecolors[i], alpha=0.7)
        # ax.vlines(chi2_prev, ytop_prev, ybottom, color='black', linewidth=1.5, linestyle='-')

        feature = get_feature(i)
        ax.broken_barh([(prev_value, change)], (yi, h), facecolors=get_color(i), **kwargs)
        prev_value = value
    # else:
        # ax.vlines(self.chi2[-1], self.ytop[-1], self.ybottom[-1], color='black', linewidth=1.5, linestyle='-')

    ax_left2 = ax.twinx()
    ax_left2.tick_params(axis='y', direction='out', labelleft=True, labelright=False, pad=-10)
    ax_right1 = ax.twinx()
    ax_right1.tick_params(axis='y', direction='in', pad=-7)
    ax_right2 = ax.twinx()
    ax_right2.tick_params(axis='y', direction='out')
    yax_left2 = ax_left2.yaxis
    yax_right1= ax_right1.yaxis
    yax_right2= ax_right2.yaxis

    yax_list = [ax.yaxis, ax_left2.yaxis, ax_right1.yaxis, ax_right2.yaxis]
    fmt_list = ['{index}', u'{label}', '{change:+.2f}', '{value:.2f}']
    alignment = ['right', 'left', 'right', 'left']
    colorizers = [None, None, get_color, None]

    bbox = dict(alpha=0.8, color='white', fill=True, boxstyle='round', linewidth=0.0)
    def yobserver(yax):
        y1, y2 = yax.get_data_interval()
        if y2<y1:
            y1, y2 = y2, y1
        newticks, newlabels = [], [[], [], [], []]
        if ax.yaxis_inverted():
            iterator = enumerate(zip(ticks_at, reversed(labels), reversed(values), reversed(changes)))
        else:
            iterator = enumerate(zip(ticks_at, labels, values, changes))
        for i, (tick, label, value, change) in iterator:
            if tick<y1 or tick>y2:
                continue

            newticks.append(tick)
            for labelslist, fmt in zip(newlabels, fmt_list):
                labelslist.append(fmt.format(index=i, label=label, change=change, value=value))

        v1, v2 = yax.get_view_interval()

        for col, (yax, ls, align, colorizer) in enumerate(zip(yax_list, newlabels, alignment, colorizers)):
            yax.set_view_interval(v1, v2)
            yax.set_ticks(newticks, minor=False)
            ylabels = yax.set_ticklabels(ls)

            if ax.yaxis_inverted():
                label_iterator = reversed(ylabels)
            else:
                label_iterator = ylabels
            for i, label, in enumerate(label_iterator):
                label.set_bbox(bbox)
                label.set_ha(align)

                if colorizer:
                    label.set_color(get_color(i))
                label_style_fcn(col, i, values[i], changes[i], label)

    ax.yaxis.add_callback(yobserver)
    yobserver(ax.yaxis)

    # labels = ax.set_yticklabels(reversed(self.info))
    # axes = self.patch_yticklabels()
    # ax.text(0.00, -0.065, '(!0)', transform=ax.transAxes, ha='left', va='top')

class NMOSensPlotter(object):
    chi2 = None
    def __init__(self, opts):
        self.variables = opts.files[0].names
        self.size = len(opts.files)
        self.arange = np.arange(self.size)
        self.opts = opts

        self.load_data()

    def load_data(self):
        # Labels
        self.info = [data.label.decode('utf-8') for data in self.opts.files]
        self.skip  = np.array([int(data.__dict__.get('skip', 0)) for data in self.opts.files])
        self.trans = np.array([int(data.__dict__.get('transient', 0)) for data in self.opts.files])
        if self.skip[0] or self.trans[0]:
            raise Exception('Unable to skip/make transient first entry')
        skip=0
        for i in range(1, len(self.skip)):
            if skip:
                self.skip[i]+=skip
            if self.trans[i]:
                skip+=1
            else:
                skip=0

        # Chi2 values
        self.chi2_full = np.zeros(self.size+1, dtype='d')
        for i, data in enumerate(self.opts.files):
            self.chi2_full[i+1] = data.fun

        self.chi2 = self.chi2_full[1:]

        # Bar Y
        self.width=0.9
        self.hwidth=self.width*0.5

        self.yc_full = np.arange(1, -len(self.info), -1)
        self.ytop_full = self.yc_full+self.hwidth
        self.ybottom_full = self.yc_full-self.hwidth
        self.ywidth_full = self.ybottom_full-self.ytop_full

        self.yc = self.yc_full[1:]
        self.ytop = self.ytop_full[1:]
        self.ybottom = self.ybottom_full[1:]
        self.ywidth = self.ywidth_full[1:]

        # Previous idx
        self.idx_prev = np.arange(0, len(self.chi2))
        self.idx_prev -= self.skip

        # Previous step data
        self.chi2_prev = self.chi2_full[self.idx_prev]
        self.shift = self.chi2 - self.chi2_prev
        self.facecolors = [s>0 and 'green' or 'red' for s in self.shift]

        self.ytop_prev = self.ytop_full[self.idx_prev]

    def plot_nmo_sensitivity(self):
        suffix = ('nmo', 'sens')
        print('Plot NMO sensitivity')

        ax=self.figure(r'$\Delta\chi^2$')
        ax.barh(self.yc, self.chi2, self.ywidth, color=self.facecolors)
        ax.axvline(self.chi2[0], color='blue', linewidth=1, alpha=0.5)
        self.patch_yticklabels()

        self.savefig(suffix+('rel', ))

        #
        #
        #
        ax=self.figure(r'$\Delta\chi^2$')
        ax.set_xlim(self.chi2.min()-5, self.chi2.max()+1.5)

        for i in range(len(self.chi2)):
            chi2_prev = self.chi2_prev[i]
            shift = self.shift[i]
            ytop, ybottom, ytop_prev = self.ytop[i], self.ybottom[i], self.ytop_prev[i]
            ax.broken_barh([(chi2_prev, shift)], (ytop, ybottom-ytop), facecolor=self.facecolors[i], alpha=0.9)
            ax.vlines(chi2_prev, ytop_prev, ybottom, color='black', linewidth=1.5, linestyle='-')
        else:
            ax.vlines(self.chi2[-1], self.ytop[-1], self.ybottom[-1], color='black', linewidth=1.5, linestyle='-')

        axes = self.patch_yticklabels()
        ax.text(0.00, -0.065, '(!0)', transform=ax.transAxes, ha='left', va='top')

        self.savefig(suffix+('rel1', ))

        i=0
        for axis, a, b, save in self.opts.zoom_save:
            a, b = float(a), float(b)
            if axis=='x':
                for ax in axes:
                    ax.set_xlim(a, b)
            elif axis=='y':
                for ax in axes:
                    ax.set_ylim(a, b)

            if save=='1':
                self.savefig(suffix+('rel1', str(i)))
                i+=1

        ax=self.figure(r'$\Delta\chi^2$')
        ax.set_xlim(self.chi2.min()-4, self.chi2.max()+1.5)
        plt.subplots_adjust(left=0.07, right=0.90)
        ax.invert_yaxis()

        def label_style_fcn(col, i, val, change, text):
            if col!=2:
                return
            if np.fabs(change)>0.5:
                text.set_fontweight('bold')

        bars_relative(self.info, self.chi2, alpha=0.7, facecolors=('red', 'green'), label_style_fcn=label_style_fcn)

        # ax=self.figure(r'$\Delta\chi^2$')
        # ax.barh(self.info, self.shift, color=self.facecolors)
        # ax.set_ylim(reversed(ax.get_ylim()))
        # self.savefig(suffix+('relshift', ))

    def plot(self):
        self.plot_nmo_sensitivity()

        if self.opts.show:
            plt.show()

    def figure(self, ylabel):
        fig = plt.figure(figsize=self.opts.figsize)
        ax = plt.subplot(111, xlabel=ylabel, title='NMO sensitivity')
        ax.minorticks_on()
        ax.tick_params(axis='y', direction='inout', which='minor', left=False, right=False, length=0.0)
        ax.grid(axis='x')

        plt.subplots_adjust(left=0.3)

        formatter = ax.xaxis.get_major_formatter()
        formatter.set_useOffset(False)
        formatter.set_powerlimits((-2,2))
        formatter.useMathText=True

        plt.subplots_adjust(left=0.2)

        return ax

    def patch_yticklabels(self, ax=None):
        ax = ax or plt.gca()
        ax.set_ylim(-len(self.info)+0.5, 0.5)
        ax.tick_params(axis='y', which='major', direction='in', pad=-10)
        plt.yticks(sorted(range(0, -len(self.info), -1)))
        labels = ax.set_yticklabels(reversed(self.info))
        ax_right1 = ax.twinx()
        # ax_right1.set_ylabel(r'$\Delta\chi^2$')
        plt.tick_params(axis='y', direction='in', pad=-7)
        ax_right2 = ax.twinx()
        plt.tick_params(axis='y', direction='out')

        ax_left2 = ax.twinx()
        plt.tick_params(axis='y', direction='out', labelleft=True, labelright=False)

        bbox_left  = dict(alpha=0.8, color='white', fill=True, boxstyle='round', linewidth=0.0)
        bbox_right = dict(alpha=0.8, color='white', fill=True, boxstyle='round', linewidth=0.0)
        for label in ax.get_yticklabels():
            label.set_bbox(bbox_left)
            label.set_ha('left')

        plt.subplots_adjust(left=0.07, right=0.90)

        ax_left2.set_ylim(*ax.get_ylim())
        ax_left2.set_yticks(ax.get_yticks())
        labels = ax_left2.set_yticklabels(['{: 2d}'.format(i) for i in reversed(range(len(self.shift)))])

        ax_right1.set_ylim(*ax.get_ylim())
        ax_right1.set_yticks(ax.get_yticks())
        labels = [u'{:+.2g}'.format(c).replace(u'-',u'–') for c in reversed(self.shift)]
        labels[-1]=''
        labels = ax_right1.set_yticklabels(labels)
        for label, fc, shift in zip(labels, reversed(self.facecolors), reversed(self.shift)):
            label.set_bbox(bbox_right)
            label.set_ha('right')
            label.set_color(fc)
            if np.fabs(shift)>0.5:
                label.set_fontweight('bold')
            label.set_zorder(-1)

        ax_right2.set_ylim(*ax.get_ylim())
        ax_right2.set_yticks(ax.get_yticks())
        labels = ax_right2.set_yticklabels(['{:.2f}'.format(c) for c in reversed(self.chi2)])
        for label in labels:
            label.set_bbox(bbox_right)
            label.set_ha('left')

        if self.opts.lines:
            xlim = ax.get_xlim()
            linesy = -np.array(self.opts.lines)+0.5
            ax.hlines(linesy, xlim[0], xlim[1], linestyle='dashed', linewidth=1.0, alpha=0.6)
            ax.set_xlim(*xlim)

        return ax, ax_left2, ax_right1, ax_right2

    def savefig(self, suffix):
        savefig(self.opts.output, dpi=300, suffix=suffix)

if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser(description=__doc__)
    def loader(name):
        with open(name, 'r') as input:
            data = load(input, Loader)
            return Namespace(**data)

    parser.add_argument('files', nargs='+', help='Yaml file to load', type=loader)
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('-s', '--show', action='store_true', help='show figures')
    parser.add_argument('-l', '--lines', type=int, nargs='+', default=[], help='add separator lines after values')
    parser.add_argument('--zoom-save', '--zs', nargs=4, default=[], action='append')
    parser.add_argument('--figsize', nargs=2, type=float, help='figsize')

    plotter=NMOSensPlotter(parser.parse_args())
    plotter.plot()
