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
        chi2 = np.zeros(self.size, dtype='d')

        for i, data in enumerate(self.opts.files):
            chi2[i] = data.fun

        return chi2

    def figure(self, ylabel):
        fig = plt.figure()
        ax = plt.subplot(111, ylabel='Iteration', xlabel=ylabel, title='NMO sensitivity')
        ax.minorticks_on()
        ax.grid(axis='x')

        plt.subplots_adjust(left=0.2)

        formatter = ax.xaxis.get_major_formatter()
        formatter.set_useOffset(False)
        formatter.set_powerlimits((-2,2))
        formatter.useMathText=True

        return ax

    def savefig(self, suffix):
        savefig(self.opts.output, dpi=300, suffix=suffix)

    def plot_nmo_sensitivity(self):
        suffix = ('nmo', 'sens')
        print('Plot NMO sensitivity')

        chi2 = self.get_chi2()
        shift = np.zeros_like(chi2)
        shift[1:] = chi2[1:] - chi2[:-1]

        ax=self.figure(r'$\chi^2$')
        ax.barh(self.info, chi2)
        ax.set_ylim(reversed(ax.get_ylim()))

        self.savefig(suffix+('rel', ))

        ax=self.figure(r'$\chi^2$')

        prev = 0
        for i, (label, c) in enumerate(zip(self.info, chi2)):
            ax.broken_barh([(prev, c-prev)], (-i, 0.9))
            prev = c

        ax.set_ylim(-len(self.info), 0)
        ax.set_yticks(range(-len(self.info), 0))
        ax.set_yticklabels(reversed(self.info))

        self.savefig(suffix+('rel1', ))

        ax=self.figure(r'$\Delta\chi^2$')
        ax.barh(self.info, shift)
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
