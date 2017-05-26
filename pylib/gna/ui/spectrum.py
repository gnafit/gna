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

        parser.add_argument('-dp', '--difference_plot', default=[],
                            action=append_typed(observable),
                            help='Subtract two obs, they MUST have the same binning')
        parser.add_argument('-p', '--plot', default=[],
                            metavar=('DATA',),
                            action=append_typed(observable))
        parser.add_argument('--plot_type', choices=['histo', 'bin_center'],
                            default='bin_center', metavar='PLOT_TYPE',
                            help='Select plot type')
        parser.add_argument('-l', '--legend', action='append', default=[],
                            metavar=('Legend',),
                            help='Add legend to the plot, note that number of legends must match the number of plots')
        parser.add_argument('--plot_kwargs', type=yaml.load,
                            help='All additional plotting options go here. They are applied for all plots')
        parser.add_argument('--savefig', default='', help='Path to save figure')


    def run(self):
        self.maps = {'bin_center': self.make_bin_center, 'histo': self.make_histo}

        self.legends = self.opts.legend
        if self.legends:
            show_legend = True
        else:
            show_legend = False

        plot_kwargs = self.opts.plot_kwargs if self.opts.plot_kwargs else {}

        self.data_storage = [obs.data() for obs in self.opts.plot]
        self.edges_storage = [np.array(obs.datatype().hist().edges()) for obs in self.opts.plot]

        if self.opts.difference_plot:
            self.make_diff()

        while len(self.data_storage) > len(self.legends):
            print "Amount of data and amount of data doesn't match. Perhaps it is not what you want. Filling legend with empty strings"
            self.legends.append('')

        ax = plt.gca()

        for data, edges, legend in zip(self.data_storage, self.edges_storage, self.legends):
            if (edges.shape[0]-1,) != data.shape:
                msg = "edges shape mismatch for 1d histogram: edges {0!r} vs values {1!r}"
                raise Exception(msg.format((edges.shape,), data.shape))
            x, y = self.maps[self.opts.plot_type](edges, data)
            ax.plot(x, y, label=legend, **plot_kwargs)

        if show_legend:
            ax.legend(loc='best')

        if self.opts.savefig:
            plt.savefig(self.opts.savefig)
        else:
            plt.show()

    def make_bin_center(self, edges, data):
        return (edges[:-1] + edges[1:])/2, data

    def make_histo(self, edges, data):
        zero_value =  0.0
        y = np.empty(len(data)*2+2)
        y[0], y[-1]=zero_value, zero_value
        y[1:-1] = np.vstack((data, data)).ravel(order='F')
        x = np.vstack((edges, edges)).ravel(order='F')
        return x, y

    def make_diff(self):
            diff = self.opts.difference_plot
            data = diff[0].data() - diff[1].data()
            self.data_storage.append(data)
            edges_0 = np.array(diff[0].datatype().hist().edges())
            edges_1 = np.array(diff[1].datatype().hist().edges())
            if not np.array_equal(edges_0, edges_1):
                raise Exception("You subtract histos with different binning")
            self.edges_storage.append(edges_0)

