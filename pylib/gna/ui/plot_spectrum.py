"""Plot 1d ovservables"""

from gna.ui import basecmd, append_typed, qualified
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_tools.helpers import savefig
import numpy as np
import yaml
from gna.bindings import common
from gna.env import PartNotFoundError, env

class cmd(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

        if self.opts.plot_type:
            if self.opts.vs and self.opts.plot_type!='plot':
                print('\033[35mWarning! plot-type option was reset to "plot"')
                self.opts.plot_type='plot'
        elif self.opts.vs:
            self.opts.plot_type='plot'
        else:
            self.opts.plot_type='bin_center'

    @classmethod
    def initparser(cls, parser, env):
        def observable(path):
            try:
                return env.ns('').getobservable(path)
            except KeyError:
                raise PartNotFoundError("observable", path)

        def yamlload(s):
            ret = yaml.load(s, Loader=yaml.Loader)
            return ret

        what = parser.add_mutually_exclusive_group()
        what.add_argument('-dp', '--difference-plot', '--diff', default=[], nargs=2, action=append_typed(observable, observable), help='Subtract two obs, they MUST have the same binning')
        what.add_argument('--ratio',                            default=[], nargs=2, action=append_typed(observable, observable), help="Plot ratio of 2 observables")
        what.add_argument('--offset-ratio', '--oratio',         default=[], nargs=2, action=append_typed(observable, observable), help="Plot ratio of 2 observables -1")
        what.add_argument('-p', '--plot', default=[], metavar=('DATA',), action=append_typed(observable))

        parser.add_argument('--vs', metavar='X points', type=observable, help='Points over X axis to plot vs')
        parser.add_argument('--plot-type', choices=['histo', 'bin_center', 'bar', 'hist', 'errorbar', 'plot'], metavar='PLOT_TYPE',
                            help='Select plot type')
        parser.add_argument('--scale', action='store_true', help='scale histogram by bin width')
        parser.add_argument('-l', '--legend', action='append', default=[],
                            metavar=('Legend',),
                            help='Add legend to the plot, note that number of legends must match the number of plots')
        parser.add_argument('--plot-kwargs', type=yamlload,
                            help='All additional plotting options go here. They are applied for all plots')

    def run(self):
        self.legends = self.opts.legend
        if self.legends:
            show_legend = True
        else:
            show_legend = False

        plot_kwargs = self.opts.plot_kwargs if self.opts.plot_kwargs else {}
        if self.opts.scale:
            plot_kwargs.setdefault('scale', 'width')

        self.data_storage = [obs.data() for obs in self.opts.plot]
        self.edges_storage = [np.array(obs.datatype().hist().edges()) for obs in self.opts.plot]

        while len(self.data_storage) > len(self.legends):
            print "Amount of data and amount of data doesn't match. Perhaps it is not what you want. Filling legend with empty strings"
            self.legends.append('')

        for output, legend in zip(self.opts.plot, self.legends):
            if self.opts.plot_type=='bar':
                output.plot_bar(label=legend, **plot_kwargs)
            elif self.opts.plot_type in ('histo', 'hist'):
                output.plot_hist(label=legend, **plot_kwargs)
            elif self.opts.plot_type=='errorbar':
                output.plot_errorbar(yerr='stat', label=legend, **plot_kwargs)
            elif self.opts.plot_type=='plot':
                output.plot_vs(self.opts.vs, label=legend, **plot_kwargs)
            else:
                output.plot_hist_centers(label=legend, **plot_kwargs)

        legends = self.legends[len(self.opts.plot):]
        if self.opts.difference_plot:
            for ((output1, output2), label) in zip(self.opts.difference_plot, legends):
                output1.plot_hist(diff=output2, label=label, **plot_kwargs)
        if self.opts.ratio:
            for ((output1, output2), label) in zip(self.opts.difference_plot, legends):
                output1.plot_hist(ratio=output2, label=label, **plot_kwargs)
        if self.opts.offset_ratio:
            for ((output1, output2), label) in zip(self.opts.difference_plot, legends):
                output1.plot_hist(offsetratio=output2, label=label, **plot_kwargs)

        if show_legend:
            ax = plt.gca()
            ax.legend(loc='best')

def list_get(lst, idx, default):
    return lst[idx] if idx<len(lst) else default
