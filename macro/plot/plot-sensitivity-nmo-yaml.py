#!/usr/bin/env python

"""Read a set of yaml files with fit results and plot sensitivity to neutrino mass hierachy (NMO)

It expects the fit data: the model (IO) fit against asimov prediction of model with NO hypothesis

"""

from __future__ import print_function
from matplotlib import pyplot as plt
from yaml import load, FullLoader
from mpl_tools.helpers import savefig
import numpy as np

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
        self.info = [data.label for data in self.opts.files]

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
        print('idx prev before', self.idx_prev)
        for cur, prev in self.opts.previous:
            assert cur!=0
            self.idx_prev[cur]=prev+1
        print('idx prev after', self.idx_prev)

        # Previous step data
        self.chi2_prev = self.chi2_full[self.idx_prev]
        self.shift = self.chi2 - self.chi2_prev
        self.facecolors = [s>0 and 'green' or 'red' for s in self.shift]

        self.ytop_prev = self.ytop_full[self.idx_prev]

        print('ytop', self.ytop)
        print('ytop_prev', self.ytop_prev)
        print('ybottom', self.ybottom)

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

        for i in range(len(self.chi2)):
            chi2_prev = self.chi2_prev[i]
            shift = self.shift[i]
            ytop, ybottom, ytop_prev = self.ytop[i], self.ybottom[i], self.ytop_prev[i]
            ax.broken_barh([(chi2_prev, shift)], (ytop, ybottom-ytop), facecolor=self.facecolors[i], alpha=0.7)
            ax.vlines(chi2_prev, ytop_prev, ybottom, color='black', linewidth=1.5, linestyle='-')
        else:
            ax.vlines(self.chi2[-1], self.ytop[-1], self.ybottom[-1], color='black', linewidth=1.5, linestyle='-')

        self.patch_yticklabels()

        self.savefig(suffix+('rel1', ))

        ax.set_xlim(left=self.chi2.min()-3)
        self.savefig(suffix+('rel1', ))

        # #
        # #
        # #
        # ax=self.figure(r'$\Delta\chi^2$')
        # ax.barh(self.info, self.shift, color=self.facecolors)
        # ax.set_ylim(reversed(ax.get_ylim()))
        # self.savefig(suffix+('relshift', ))

    def plot(self):
        self.plot_nmo_sensitivity()

        if self.opts.show:
            plt.show()

    def figure(self, ylabel):
        fig = plt.figure()
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
        ax1 = ax.twinx()
        ax1.set_ylabel(r'$\Delta\chi^2$')
        plt.tick_params(axis='y', direction='in', pad=-7)

        bbox_left  = dict(alpha=0.8, color='white', fill=True, boxstyle='round', linewidth=0.0)
        bbox_right = dict(alpha=0.8, color='white', fill=True, boxstyle='round', linewidth=0.0)
        for label in ax.get_yticklabels():
            label.set_bbox(bbox_left)
            label.set_ha('left')

        plt.subplots_adjust(left=0.05, right=0.95)

        ax1.set_ylim(*ax.get_ylim())
        ax1.set_yticks(ax.get_yticks())
        ax1.set_yticks(ax.get_yticks())
        labels = ax1.set_yticklabels(['{:.2f}'.format(c) for c in reversed(self.chi2)])

        for label in labels:
            label.set_bbox(bbox_right)
            label.set_ha('right')

        if self.opts.lines:
            xlim = ax.get_xlim()
            linesy = -np.array(self.opts.lines)+0.5
            ax.hlines(linesy, xlim[0], xlim[1], linestyle='dashed', linewidth=1.0, alpha=0.6)
            ax.set_xlim(*xlim)

    def savefig(self, suffix):
        savefig(self.opts.output, dpi=300, suffix=suffix)


if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser(description=__doc__)
    def loader(name):
        with open(name, 'r') as input:
            data = load(input, FullLoader)
            return Namespace(**data)

    parser.add_argument('files', nargs='+', help='Yaml file to load', type=loader)
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('-s', '--show', action='store_true', help='show figures')
    parser.add_argument('-p', '--previous', type=int, default=[], nargs=2, action='append', help='previous value to calculate diff')
    parser.add_argument('-l', '--lines', type=int, nargs='+', default=[], help='add separator lines after values')

    plotter=NMOSensPlotter(parser.parse_args())
    plotter.plot()
