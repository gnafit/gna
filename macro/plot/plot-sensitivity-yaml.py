#!/usr/bin/env python
# encoding: utf-8

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

        # # Chi2 values
        # self.chi2_full = np.zeros(self.size+1, dtype='d')
        # for i, data in enumerate(self.opts.files):
            # self.chi2_full[i+1] = data.fun

        # self.chi2 = self.chi2_full[1:]

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
        self.idx_prev = np.arange(0, len(self.info))
        self.idx_prev -= self.skip

        # # Previous step data
        # self.chi2_prev = self.chi2_full[self.idx_prev]
        # self.shift = self.chi2 - self.chi2_prev
        # self.facecolors = [s>0 and 'green' or 'red' for s in self.shift]

        self.ytop_prev = self.ytop_full[self.idx_prev]


    def plot(self):
        for name in self.variables:
            self.plot_variable(name)

            # break

        if self.opts.show:
            plt.show()

    def plot_variable(self, varname):
        suffix = tuple(varname.split('.'))
        print('Plot variable', varname)

        try:
            centrals, errs, shift, facecolors = self.get_errors(varname)
            relerr = 100.0*errs/centrals
            print(centrals)
            print(errs)
            print(relerr)
        except:
            return

        # fig = plt.figure()
        # ax = plt.subplot(111, xlabel='Iteration', ylabel='Result', title=varname)
        # ax.minorticks_on()
        # ax.grid()

        # ebopts=dict(fmt='o', markerfacecolor='none')
        # ax.errorbar(self.arange, centrals, errs, **ebopts)
        #
        ax=self.figure(varname, 'Absolute error')
        ax.barh(self.yc, errs, self.ywidth, color=facecolors)
        self.patch_yticklabels()
        self.savefig(suffix+('abs', ))

        ax=self.figure(varname, 'Relative error, %')
        ax.barh(self.yc, relerr, self.ywidth, color=facecolors)
        self.patch_yticklabels()

        self.savefig(suffix+('rel', ))

        # ax=self.figure(varname, 'Relative error offset, %')
        # ax.barh(self.info, 100.0*shift/centrals)

        # self.savefig(suffix+('relshift', ))

    def get_errors(self, varname):
        centrals_full = np.zeros(self.size+1, dtype='d')
        errs_full     = centrals_full.copy()

        for i, data in enumerate(self.opts.files):
            centrals_full[i+1] = data.xdict[varname]
            errs_full[i+1]     = data.errorsdict[varname]

        centrals_full[0] = centrals_full[1]

        errs_prev = errs_full[self.idx_prev]
        shift = errs_full[1:] - errs_prev
        facecolors = [s<=0 and 'green' or 'red' for s in shift]

        return centrals_full[1:], errs_full[1:], shift, facecolors

    def figure(self, varname, ylabel):
        fig = plt.figure(figsize=self.opts.figsize)
        ax = plt.subplot(111, xlabel=ylabel, title=varname)
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
        # ax1 = ax.twinx()
        # ax1.set_ylabel(r'$\Delta\chi^2$')
        # plt.tick_params(axis='y', direction='in', pad=-7)
        # ax2 = ax.twinx()
        # plt.tick_params(axis='y', direction='out')

        bbox_left  = dict(alpha=0.8, color='white', fill=True, boxstyle='round', linewidth=0.0)
        bbox_right = dict(alpha=0.8, color='white', fill=True, boxstyle='round', linewidth=0.0)
        for label in ax.get_yticklabels():
            label.set_bbox(bbox_left)
            label.set_ha('left')

        plt.subplots_adjust(left=0.02, right=0.90)

        # ax1.set_ylim(*ax.get_ylim())
        # ax1.set_yticks(ax.get_yticks())
        # labels = [u'{:+.2g}'.format(c).replace(u'-',u'–') for c in reversed(self.shift)]
        # labels[-1]=''
        # labels = ax1.set_yticklabels(labels)
        # for label, fc, shift in zip(labels, reversed(self.facecolors), reversed(self.shift)):
            # label.set_bbox(bbox_right)
            # label.set_ha('right')
            # label.set_color(fc)
            # if np.fabs(shift)>0.5:
                # label.set_fontweight('bold')

        # ax2.set_ylim(*ax.get_ylim())
        # ax2.set_yticks(ax.get_yticks())
        # labels = ax2.set_yticklabels(['{:.2f}'.format(c) for c in reversed(self.chi2)])
        # for label in labels:
            # label.set_bbox(bbox_right)
            # label.set_ha('left')

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
    parser.add_argument('-l', '--lines', type=int, nargs='+', default=[], help='add separator lines after values')
    parser.add_argument('--figsize', nargs=2, type=float, help='figsize')

    plotter=UncertaintyPlotter(parser.parse_args())
    plotter.plot()
