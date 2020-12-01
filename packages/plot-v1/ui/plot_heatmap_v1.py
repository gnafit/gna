"""Plot a 2-dimensional heatmap."""

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
                return env.future[path]
            except KeyError:
                import IPython; IPython.embed()
                pass

            # To be deprecated spectra location
            try:
                return env.ns('').getobservable(path)
            except KeyError:
                raise PartNotFoundError("observable", path)

        def yamlload(s):
            ret = yaml.load(s, Loader=yaml.Loader)
            return ret

        parser.add_argument('plot', nargs='+', metavar='DATA', default=[], action=append_typed(observable))
        parser.add_argument('--plot-kwargs', type=yamlload,
                            help='All additional plotting options go here. They are applied for all plots')
        parser.add_argument('-l', '--log', action='store_true', help='use log scale')

        parse_filters = parser.add_mutually_exclusive_group()
        parse_filters.add_argument('--filter', '-f', choices=filters.keys(), help='filter the matrix')
        # parse_filters.add_argument('--rebin', type=int, help='rebin the matrix N times')

    def run(self):
        plot_kwargs = self.opts.plot_kwargs if self.opts.plot_kwargs else {}
        plot_kwargs.setdefault('mask', 0.0)
        plot_kwargs.setdefault('colorbar', True)
        if self.opts.filter:
            plot_kwargs['preprocess'] = filters[self.opts.filter]
        # if self.opts.rebin:
            # plot_kwargs['preprocess'] = make_rebinner(self.opts.filter)
        if self.opts.log:
            plot_kwargs['norm'] = colors.LogNorm()

        for (obs,) in self.opts.plot:
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

cmd.__tldr__ =  """\
                The module plots a 2-dimensional output as a heatmap.

                Plot a lower triangular matrix L — the Cholesky decomposition of the covariance matrix:
                ```sh
                ./gna \\
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
                    -- pargroup minpars peak_f -vv -m free \\
                    -- pargroup covpars peak_f -vv -m constrained \\
                    -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum -vv \\
                    -- analysis-v1 peak --datasets peak -p covpars -v \\
                    -- env-print analysis \\
                    -- plot-heatmap-v1 analysis.peak.0.L -f tril \\
                    -- mpl-v1 --xlabel columns --ylabel rows -t 'Cholesky decomposition, L' -s
                ```
                Here the filter 'tril' provided via `-f` ensures that only the lower triangular is plotted since
                it is not guaranteed that the upper matrix is reset to zero.

                For more details on decorations and saving see `mpl-v1`.

                See also: `mpl-v1`, `plot-spectrum-v1`.
                """
