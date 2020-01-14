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

        self.info = [data.label for data in opts.files]

    def plot(self):
        self.plot_nmo_sensitivity()

        if self.opts.show:
            plt.show()

    def get_chi2(self):
        if self.chi2 is not None:
            return self.chi2

        self.chi2 = np.zeros(self.size, dtype='d')

        for i, data in enumerate(self.opts.files):
            self.chi2[i] = data.fun

        return self.chi2

    def figure(self, ylabel):
        fig = plt.figure()
        ax = plt.subplot(111, xlabel=ylabel, title='NMO sensitivity')
        ax.minorticks_on()
        ax.grid(axis='x')

        plt.subplots_adjust(left=0.3)
        plt.tick_params(axis='y', which='minor', left=False)

        formatter = ax.xaxis.get_major_formatter()
        formatter.set_useOffset(False)
        formatter.set_powerlimits((-2,2))
        formatter.useMathText=True

        plt.subplots_adjust(left=0.2)

        return ax

    def patch_yticklabels(self, ax=None):
        ax = ax or plt.gca()
        plt.tick_params(axis='y', direction='in', pad=-10)
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
        labels = ax1.set_yticklabels(['{:.2f}'.format(c) for c in reversed(self.get_chi2())])

        for label in labels:
            label.set_bbox(bbox_right)
            label.set_ha('right')

    def savefig(self, suffix):
        savefig(self.opts.output, dpi=300, suffix=suffix)

    def plot_nmo_sensitivity(self):
        suffix = ('nmo', 'sens')
        print('Plot NMO sensitivity')

        chi2 = self.get_chi2()
        shift = np.zeros_like(chi2)
        shift[0] = chi2[0]
        shift[1:] = chi2[1:] - chi2[:-1]
        facecolors = [s>0 and 'green' or 'red' for s in shift]

        ax=self.figure(r'$\Delta\chi^2$')
        ax.barh(self.info, chi2, color=facecolors)
        ax.set_ylim(reversed(ax.get_ylim()))
        ax.axvline(chi2[0], color='blue', linewidth=1, alpha=0.5)
        self.patch_yticklabels()

        self.savefig(suffix+('rel', ))

        #
        #
        #
        ax=self.figure(r'$\Delta\chi^2$')

        prev, y_prev = 0, 0.0
        markwidth = 0.10
        for i, (label, c, color) in enumerate(zip(self.info, chi2, facecolors)):
            offset = c-prev
            y1, h = -i+0.4, -0.8
            y2 = y1+h
            ax.broken_barh([(prev, offset)], (y1, h), facecolor=color, alpha=0.7) #, alpha=0.5)
            # if offset>0:
                # ax.broken_barh([(c-markwidth, markwidth)], (-i+0.4, -0.8), facecolor=color)
            # else:
                # ax.broken_barh([(c, markwidth)], (-i+0.4, -0.8), facecolor=color)
                #

            if i:
                ax.vlines(prev, y_prev, y2, color='black', linewidth=1.5, linestyle='-')

            y_prev = y1
            prev = c
        else:
            ax.vlines(prev, y_prev, y2, color='black', linewidth=1.5, linestyle='-')

        ax.set_ylim(-len(self.info)+0.5, 0.5)
        plt.yticks(sorted(range(0, -len(self.info), -1)), reversed(self.info))
        self.patch_yticklabels()

        self.savefig(suffix+('rel1', ))

        ax.set_xlim(left=chi2.min()-3)
        self.savefig(suffix+('rel1', ))

        #
        #
        #
        ax=self.figure(r'$\Delta\chi^2$')
        ax.barh(self.info, shift, color=facecolors)
        ax.set_ylim(reversed(ax.get_ylim()))

        self.savefig(suffix+('relshift', ))

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

    plotter=NMOSensPlotter(parser.parse_args())
    plotter.plot()
