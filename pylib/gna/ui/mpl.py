# encoding: utf-8

u"""Change global parameters of the matplotlib"""

from gna.ui import basecmd
import ROOT
import numpy as np
from gna import constructors as C
import yaml
import pprint

def yaml_load(s):
    return yaml.load(s, Loader=yaml.Loader) or {}

undefined=object()

class cmd(basecmd):
    _ax = None
    _fig = None
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-v', '--verbose', action='count', help='verbosity level')

        mpl = parser.add_argument_group(title='matplotlib', description='General matplotlib parameters')

        inter = mpl.add_mutually_exclusive_group()
        inter.add_argument( '-i', '--interactive', action='store_true', help='switch to interactive matplotlib' )
        inter.add_argument('-b', '--batch', action='store_true', help='run in batch mode')

        mpl.add_argument('-l', '--latex', action='store_true', help='enable latex mode')
        mpl.add_argument('-r', '--rcparam', '--rc', nargs='+', default=[], type=yaml_load, help='YAML dictionary with RC configuration')

        fig = parser.add_argument_group(title='figure', description='Figure modification parameters')
        fig.add_argument('-f', '--figure', nargs='?', default=undefined, type=yaml_load, help='create new figure', metavar='kwargs')

        axis = parser.add_argument_group(title='axis', description='Axis modification parameters')
        axis.add_argument('-t', '--title', help='axis title')
        axis.add_argument('--xlabel', '--xl', help='x label')
        axis.add_argument('--ylabel', '--yl', help='y label')
        axis.add_argument('--ylim', nargs='+', type=float, help='Y limits')
        axis.add_argument('--xlim', nargs='+', type=float, help='X limits')
        axis.add_argument('--legend', nargs='?', default=undefined, help='legend (optional: position)')
        axis.add_argument('--scale', nargs=2, help='axis scale', metavar=('axis', 'scale'))

        axis.add_argument('-g', '--grid', action='store_true', help='draw grid')
        axis.add_argument('--minor-ticks', '--mt', action='store_true', help='minor ticks')
        axis.add_argument('--ticks-extra', '--te', nargs='+', default=[], help='Add extra ticks', metavar=('axis', 'tick1'))

        fig1 = parser.add_argument_group(title='figure (after)', description='Figure parameters (after)')
        fig1.add_argument('-o', '--output', '--savefig', nargs='+', help='path to save figure')
        fig1.add_argument('-s', '--show', action='store_true', help='show the figure')

    def init(self):
        self.configure_mpl()
        self.configure_figure()
        self.configure_axis()
        self.configure_figure1()

    @property
    def ax(self):
        if self._ax is None:
            from matplotlib import pyplot as plt
            self._ax = plt.gca()
        return self._ax

    @property
    def fig(self):
        if self._fig is None:
            from matplotlib import pyplot as plt
            self._fig = plt.gcf()
        return self._fig

    def configure_mpl(self):
        import matplotlib as mpl
        if self.opts.batch:
            if self.opts.verbose:
                print('Batch mode matplotlib')
            mpl.use('Agg', force=True)

        if self.opts.interactive:
            if self.opts.verbose:
                print('Interactive matplotlib')
            mpl.pyplot.ion()

        if self.opts.latex:
            if self.opts.verbose:
                print('Matplotlib with latex')
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.unicode'] = True
            mpl.rcParams['font.size'] = 13

        if self.opts.verbose>1 and self.opts.rcparam:
            print('Matplotlib extra options')
            for d in self.opts.rcparam:
                pprint.pprint(d)

        for d in self.opts.rcparam:
            mpl.rcParams.update(d)

    def configure_figure(self):
        from matplotlib import pyplot as plt

        if self.opts.figure is not undefined:
            if not self.opts.figure:
                self.opts.figure={}
            plt.figure(**self.opts.figure)

    def configure_figure1(self):
        from matplotlib import pyplot as plt
        if self.opts.output:
            from mpl_tools.helpers import savefig
            savefig(self.opts.output)

        if self.opts.show:
            plt.show()

    def configure_axis(self):
        if self.opts.title:
            self.ax.set_title(self.opts.title)

        if self.opts.xlabel:
            self.ax.set_xlabel(self.opts.xlabel)

        if self.opts.ylabel:
            self.ax.set_ylabel(self.opts.ylabel)

        if self.opts.scale:
            axis, scale = self.opts.scale
            set_scale=getattr(self.ax, 'set_{axis}scale'.format(axis=axis))
            set_scale(scale)

        if self.opts.xlim:
            self.ax.set_xlim(*self.opts.xlim)

        if self.opts.ylim:
            self.ax.set_ylim(*self.opts.ylim)

        if self.opts.grid:
            self.ax.grid()

        if self.opts.minor_ticks:
            self.ax.minorticks_on()

        if self.opts.legend is not undefined:
            if self.opts.legend:
                self.ax.legend(self.opts.legend)
            else:
                self.ax.legend()

        if len(self.opts.ticks_extra)>1:
            self.fig.canvas.draw()
            axisname, ticks = self.opts.ticks_extra[0], list(map(float, self.opts.ticks_extra[1:]))
            assert axisname in ('x', 'y'), "Unsupported axis '%s', should be 'x' or 'y'"%axisname
            axis = getattr(ax, axisname+'axis')
            axis.set_ticks(axis.get_ticklocs().tolist()+ticks)

        if self.opts.xlim:
            self.fig.canvas.draw()
            self.ax.set_xlim(*self.opts.xlim)

        if self.opts.ylim:
            self.fig.canvas.draw()
            self.ax.set_ylim(*self.opts.ylim)

