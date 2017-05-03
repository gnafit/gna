from gna.ui import basecmd, append_typed, qualified
from matplotlib import pyplot as plt
import numpy as np
import yaml

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        def observable(path):
            nspath, name = path.split('/')
            try:
                return env.ns(nspath).observables[name]
            except KeyError:
                raise PartNotFoundError("observable", path)

        parser.add_argument('-p', '--plot', default=[],
                            metavar=('DATA',),
                            action=append_typed(observable))
        parser.add_argument('--plot_type', choices=['histo', 'bin_center'],
                            default='bin_center', metavar='PLOT_TYPE',
                            help='Select plot type')
        parser.add_argument('-l', '--legend', action='append', default=[],
                            metavar=('Legend',), help='Put legends here')
        parser.add_argument('--plot_kwargs', type=yaml.load,
                            help='All additional plotting options go here. They are applied for all plots')

    def run(self):
        legends = self.opts.legend
        if legends:
            show_legend = True
        else:
            show_legend = False

        while len(self.opts.plot) != len(legends):
           legends.append('default_legend')
        ax = plt.gca()
        #  simple_kwargs = {'linestyle': '--',}
        plot_kwargs = self.opts.plot_kwargs if self.opts.plot_kwargs else {}

        for plotobj, legend in zip(self.opts.plot, legends):
            data = plotobj.data()
            edges = plotobj.datatype().hist().edges()
            if (edges.size()-1,) != data.shape:
                msg = "edges shape mismatch for 1d histogram: {0!r} vs {1!r}"
                raise Exception(msg.format((edges.size()-1,), data.shape))
            edges = np.array(edges)

            if self.opts.plot_type == 'bin_center':
                x = (edges[:-1] + edges[1:])/2
                y = data
            elif self.opts.plot_type == 'histo':
                zero_value =  0.0
                y = np.empty(len(data)*2+2)
                y[0], y[-1]=zero_value, zero_value
                y[1:-1] = np.vstack((data, data)).ravel(order='F')
                x = np.vstack((edges, edges)).ravel(order='F')

            ax.plot(x, y, label=legend, **plot_kwargs)

        if show_legend:
            ax.legend(loc='best')

        plt.show()
