from gna.ui import basecmd, append_typed, qualified
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import yaml
from gna.bindings import common
from gna.env import PartNotFoundError

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        def observable(path):
            nspath, name = path.split('/')
            try:
                return env.ns(nspath).observables[name]
            except KeyError:
                raise PartNotFoundError("observable", path)

        parser.add_argument('-dp', '--difference-plot', default=[],
                            action=append_typed(observable),
                            help='Subtract two obs, they MUST have the same binning')
        parser.add_argument('-p', '--plot', default=[],
                            metavar=('DATA',),
                            action=append_typed(observable))
        parser.add_argument('--plot-type', choices=['histo', 'bin_center', 'bar', 'hist', 'errorbar'],
                            default='bin_center', metavar='PLOT_TYPE',
                            help='Select plot type')
        parser.add_argument('--scale', action='store_true', help='scale histogram by bin width')
        parser.add_argument('-l', '--legend', action='append', default=[],
                            metavar=('Legend',),
                            help='Add legend to the plot, note that number of legends must match the number of plots')
        parser.add_argument('--plot-kwargs', type=yaml.load,
                            help='All additional plotting options go here. They are applied for all plots')
        parser.add_argument('--drawgrid', action='store_true')
        parser.add_argument('-s', '--show', action='store_true')
        parser.add_argument('--savefig', default='', help='Path to save figure')
        parser.add_argument('--title', nargs='+', help='Title to the figure')
        parser.add_argument('--new-figure', action='store_true',
                            help='Create new figure')
        parser.add_argument('--xlabel', nargs='+', required=False)
        parser.add_argument('--ylabel', nargs='+', required=False)

    def run(self):
        self.maps = {'bin_center': edges_to_centers, 'histo': edges_to_histpoints, 'bar': edges_to_barpoints}

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

        if self.opts.new_figure:
            plt.figure()
        minorLocatorx = AutoMinorLocator()
        minorLocatory = AutoMinorLocator()
        ax = plt.gca()
        if self.opts.drawgrid:
            ax.xaxis.set_minor_locator(minorLocatorx)
            ax.yaxis.set_minor_locator(minorLocatory)
            plt.tick_params(which='both', width=1)
            plt.tick_params(which='major', length=7)
            plt.tick_params(which='minor', length=4, color='k')
            ax.grid(which = 'minor', alpha = 0.3)
            ax.grid(which = 'major', alpha = 0.7)

        for output, legend in zip(self.opts.plot, self.legends):
            if self.opts.plot_type=='bar':
                output.plot_bar(label=legend, **plot_kwargs)
            elif self.opts.plot_type in ('histo', 'hist'):
                output.plot_hist(label=legend, **plot_kwargs)
            elif self.opts.plot_type=='errorbar':
                output.plot_errorbar(yerr='stat', label=legend, **plot_kwargs)
            else:
                output.plot_hist_centers(label=legend, **plot_kwargs)

        legends = self.legends[len(self.opts.plot):]
        if self.opts.difference_plot:
            output1, output2 = self.opts.difference_plot
            label = legends[0] if legends else ''
            output1.plot_hist(diff=output2, label=label, **plot_kwargs)

        if show_legend:
            ax.legend(loc='best')

        if self.opts.xlabel:
            plt.xlabel(r'{}'.format(self.opts.xlabel[0]),
                    fontsize=self.opts.xlabel[1])
        if self.opts.ylabel:
            plt.ylabel(r'{}'.format(self.opts.ylabel[0]),
                    fontsize=self.opts.xlabel[1])
        if self.opts.title:
            plt.title(r'{}'.format(self.opts.title[0]),
                    fontsize=self.opts.title[1])

        if self.opts.savefig:
            plt.savefig(self.opts.savefig)

        if self.opts.show:
            plt.show()

def edges_to_centers( edges, heights ):
    return (edges[:-1] + edges[1:])/2, heights, None

def edges_to_histpoints( edges, heights ):
    zero_value =  0.0
    y = np.empty(len(heights)*2+2)
    y[0], y[-1]=zero_value, zero_value
    y[1:-1] = np.vstack((heights, heights)).ravel(order='F')
    x = np.vstack((edges, edges)).ravel(order='F')
    return x, y, None

def edges_to_barpoints( edges, heights ):
    return edges[:-1], heights, edges[1:]-edges[:-1]
