#!/usr/bin/env python

"""Read a set of yaml files with fit results and plot sensitivity to parameters"""

from __future__ import print_function
from matplotlib import pyplot as plt
from yaml import load, FullLoader
from mpl_tools.helpers import savefig
import numpy as np

class UncertaintyPlotter(object):
    def __init__(self, opts):
        self.variables = opts.files[0].names
        self.size = len(opts.files)
        self.arange = np.arange(self.size)
        self.opts = opts

    def plot(self):
        for name in self.variables:
            self.plot_variable(name)

        if self.opts.show:
            plt.show()

    def get_errors(self, varname):
        centrals = np.zeros(self.size, dtype='d')
        errs     = centrals.copy()

        for i, data in enumerate(self.opts.files):
            centrals[i] = data.xdict[varname]
            errs[i]     = data.errorsdict[varname]

        return centrals, errs

    def plot_variable(self, varname):
        print('Plot variable', varname)

        centrals, errs = self.get_errors(varname)

        # fig = plt.figure()
        # ax = plt.subplot(111, xlabel='Iteration', ylabel='Result', title=varname)
        # ax.minorticks_on()
        # ax.grid()

        # ebopts=dict(fmt='o', markerfacecolor='none')
        # ax.errorbar(self.arange, centrals, errs, **ebopts)

        fig = plt.figure()
        ax = plt.subplot(111, xlabel='Iteration', ylabel='Relative error, %', title=varname)
        ax.minorticks_on()
        ax.grid(axis='x')

        ax.barh([str(i) for i in self.arange], 100.0*errs/centrals)

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

    plotter=UncertaintyPlotter(parser.parse_args())
    plotter.plot()
