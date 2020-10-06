"""Plot 2d heatmaps"""

from gna.ui import basecmd, append_typed, qualified
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import yaml
from gna.bindings import common
from gna.env import PartNotFoundError, env
import matplotlib.colors as colors

filters = {}

class cmd(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    @classmethod
    def initparser(cls, parser, env):
        def observable(path):
            #
            # Future spectra location
            #
            try:
                return env.future['spectra'][path]
            except KeyError:
                pass

            # To be deprecated spectra location
            try:
                return env.ns('').getobservable(path)
            except KeyError:
                raise PartNotFoundError("observable", path)

        def yamlload(s):
            ret = yaml.load(s, Loader=yaml.Loader)
            return ret

        parser.add_argument('-p', '--plot', default=[], metavar=('DATA',), action=append_typed(observable))
        parser.add_argument('--plot-kwargs', type=yamlload,
                            help='All additional plotting options go here. They are applied for all plots')
        parser.add_argument('--filter', '-f', choices=filters.keys(), help='filter the matrix')
        parser.add_argument('-l', '--log', action='store_true', help='use log scale')

    def run(self):
        plot_kwargs = self.opts.plot_kwargs if self.opts.plot_kwargs else {}
        plot_kwargs.setdefault('mask', 0.0)
        plot_kwargs.setdefault('colorbar', True)
        if self.opts.filter:
            plot_kwargs['preprocess'] = filters[self.opts.filter]
        if self.opts.log:
            plot_kwargs['norm'] = colors.LogNorm()

        for obs in self.opts.plot:
            obs.plot_matshow(**plot_kwargs)

def triu(buf):
    return np.triu(buf, 1)

def tril(buf):
    return np.tril(buf, -1)

def diag(buf):
    return np.diag(np.diag(buf))

filters['triu'] = triu
filters['tril'] = tril
filters['diag'] = diag
