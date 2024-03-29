"""Change global parameters of the matplotlib, decorate figures and save images."""

import pprint
import yaml
from gna.ui import basecmd
from env.lib.cwd import update_namespace_cwd

def yaml_load(s):
    return yaml.load(s, Loader=yaml.Loader) or {}

class cmd(basecmd):
    _ax = None
    _fig = None
    undefined=object()
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity level')

        mpl = parser.add_argument_group(title='matplotlib', description='General matplotlib parameters')

        inter = mpl.add_mutually_exclusive_group()
        inter.add_argument( '-i', '--interactive', action='store_true', help='switch to interactive matplotlib' )
        inter.add_argument('-b', '--batch', action='store_true', help='run in batch mode')

        mpl.add_argument('-l', '--latex', action='store_true', help='enable latex mode')
        mpl.add_argument('-r', '--rcparam', '--rc', nargs='+', default=[], type=yaml_load, help='YAML dictionary with RC configuration')
        mpl.add_argument('--style', default=None, help='load matplotlib style')

        fig = parser.add_argument_group(title='figure', description='Figure modification parameters')
        fig.add_argument('-f', '--figure', nargs='?', default=cls.undefined, type=yaml_load, help='create new figure', metavar='kwargs')

        axis = parser.add_argument_group(title='axis', description='Axis modification parameters')
        axis.add_argument('-t', '--title', help='axis title')
        axis.add_argument('--xlabel', '--xl', help='x label')
        axis.add_argument('--ylabel', '--yl', help='y label')
        axis.add_argument('--ylim', nargs='+', type=float, help='Y limits')
        axis.add_argument('--xlim', nargs='+', type=float, help='X limits')
        axis.add_argument('--legend', nargs='?', default=cls.undefined, type=yaml_load, help='legend (optional: kwargs)')
        axis.add_argument('--scale', nargs=2, help='axis scale', metavar=('axis', 'scale'))
        axis.add_argument('--powerlimits', '--pl', nargs=3, help='set scale pwerlimits', metavar=('axis', 'pmin', 'pmax'))
        axis.add_argument('--step-color', nargs='?', default=None, const=1, type=int, help='Step color')

        axis.add_argument('-g', '--grid', action='store_true', help='draw grid')
        axis.add_argument('--minor-ticks', '--mt', action='store_true', help='minor ticks')
        axis.add_argument('--ticks-extra', '--te', nargs='+', default=[], help='Add extra ticks', metavar=('axis', 'tick1'))

        fig1 = parser.add_argument_group(title='figure (after)', description='Figure parameters (after)')
        fig1.add_argument('--tight-layout', '--tl', action='store_true', help='tight layout')
        fig1.add_argument('-o', '--output', '--savefig', nargs='+', help='path to save figure')
        fig1.add_argument('-c', '--close', action='store_true', help='close the figure')
        fig1.add_argument('-s', '--show', action='store_true', help='show the figure')

    def init(self):
        update_namespace_cwd(self.opts, 'output')
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
            self._ax = None
        return self._fig

    def configure_mpl(self):
        import matplotlib as mpl
        if self.opts.batch:
            if self.opts.verbose:
                print('Batch mode matplotlib')
            mpl.use('Agg', force=True)

        if self.opts.style:
            if self.opts.verbose:
                print(f'Load style from {self.opts.style}')
            mpl.pyplot.style.use(self.opts.style)

        if self.opts.interactive:
            if self.opts.verbose:
                print('Interactive matplotlib')
            mpl.pyplot.ion()

        if self.opts.latex:
            if self.opts.verbose:
                print('Matplotlib with latex')
            mpl.rcParams['text.usetex'] = True
            try:
                mpl.rcParams['text.latex.unicode'] = True
            except KeyError:
                pass
            mpl.rcParams['font.size'] = 13

        if self.opts.verbose>1 and self.opts.rcparam:
            print('Matplotlib extra options')
            for d in self.opts.rcparam:
                pprint.pprint(d)

        for d in self.opts.rcparam:
            mpl.rcParams.update(d)

    def configure_figure(self):
        from matplotlib import pyplot as plt

        if self.opts.figure is not self.undefined:
            if not self.opts.figure:
                self.opts.figure={}
            self._fig = plt.figure(**self.opts.figure)
            self._ax = None

    def configure_figure1(self):
        from matplotlib import pyplot as plt
        if self.opts.tight_layout:
            plt.tight_layout()
        if self.opts.output:
            from mpl_tools.helpers import savefig
            savefig(self.opts.output)

        if self.opts.show:
            plt.show()

        if self.opts.close:
            plt.close()

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

        if self.opts.powerlimits:
            axis, pmin, pmax = self.opts.powerlimits
            fmt=getattr(self.ax, '{axis}axis'.format(axis=axis)).get_major_formatter()
            fmt.set_powerlimits(tuple(map(float, (pmin, pmax))))
            fmt.useMathText=True

        if self.opts.grid:
            self.ax.grid()

        if self.opts.minor_ticks:
            self.ax.minorticks_on()

        if self.opts.legend is not self.undefined:
            legopts = self.opts.legend or {}
            self.ax.legend(**legopts)

        if len(self.opts.ticks_extra)>1:
            self.fig.canvas.draw()
            axisname, ticks = self.opts.ticks_extra[0], list(map(float, self.opts.ticks_extra[1:]))
            assert axisname in ('x', 'y'), "Unsupported axis '%s', should be 'x' or 'y'"%axisname
            axis = getattr(self.ax, axisname+'axis')
            axis.set_ticks(axis.get_ticklocs().tolist()+ticks)

        if self.opts.step_color:
            line_cycle = self.ax._get_lines
            for _ in range(self.opts.step_color):
                line_cycle.get_next_color()

        if self.opts.xlim:
            self.fig.canvas.draw()
            self.ax.set_xlim(*self.opts.xlim)

        if self.opts.ylim:
            self.fig.canvas.draw()
            self.ax.set_ylim(*self.opts.ylim)

    __tldr__ =  """\
                The module implements most of the interactions with matplotlib, excluding the plotting itself.
                When `mpl-v1` is used to produce the output files the CWD from `env-cwd` is respected.

                As the module contains a lot of options, please refer to the `gna -- mpl-v1 --help` for the reference.

                Add labels and the title:
                ```sh
                ./gna -- ... \\
                      -- mpl-v1 --xlabel 'Energy, MeV' --ylabel Entries -t 'The distribution'
                ```

                Save a figure to the 'output.pdf' and then show it:
                ```sh
                ./gna -- ... \\
                      -- mpl-v1 -o output.pdf -s \\
                ```

                Create a new figure:
                ```sh
                ./gna -- mpl-v1 -f \\
                      -- ...
                ```

                Create a new figure of a specific size:
                ```sh
                ./gna -- mpl-v1 -f '{figsize: [14, 4]}' \\
                      -- ...
                ```

                Enable latex rendering:
                ```sh
                ./gna -- mpl-v1 -l \\
                      -- ...
                ```

                `mpl-v1` enables the user to tweak RC parameters by providing YAML dictionaries with options.

                Tweak matplotlib RC parameters to make all the lines of double width and setup power limits for the tick formatter:
                ```sh
                ./gna -- mpl-v1 -r 'lines.linewidth: 2.0' 'axes.formatter.limits: [-2, 2]' \\
                      -- ...
                ```

                An example of plotting, that uses the above mentioned options:
                ```sh
                ./gna \\
                      -- env-cwd output/test-cwd \\
                      -- gaussianpeak --name peak_MC --nbins 50 \\
                      -- gaussianpeak --name peak_f  --nbins 50 \\
                      -- ns --name peak_MC --print \\
                            --set E0             values=2    fixed \\
                            --set Width          values=0.5  fixed \\
                            --set Mu             values=2000 fixed \\
                            --set BackgroundRate values=1000 fixed \\
                      -- ns --name peak_f --print \\
                            --set E0             values=2.5  relsigma=0.2 \\
                            --set Width          values=0.3  relsigma=0.2 \\
                            --set Mu             values=1500 relsigma=0.25 \\
                            --set BackgroundRate values=1100 relsigma=0.25 \\
                      -- plot-spectrum-v1 -p peak_MC.spectrum -l 'Monte-Carlo' --plot-type errorbar \\
                      -- plot-spectrum-v1 -p peak_f.spectrum -l 'Model (initial)' --plot-type hist \\
                      -- mpl-v1 --xlabel 'Energy, MeV' --ylabel entries -t 'Example plot' --grid \\
                      -- mpl-v1 -o figure.pdf -s
                ```

                See also: `plot-spectrum-v1`, `plot-heatmap-v1`, `env-cwd`.
                """
