#!/usr/bin/env python

"""Read a set of yaml files with fit results and plot sensitivity to neutrino mass hierachy (NMO)

It expects the fit data: the model (IO) fit against asimov prediction of model with NO hypothesis

"""

from matplotlib import pyplot as plt
from yaml import load, BaseLoader as Loader
from mpl_tools.helpers import savefig
import numpy as np

class Plotter(object):
    def __init__(self, opts):
	self.opts = opts

	if opts.data is not None:
	    self.chi2, self.dm = opts.data
	else:
	    self.chi2 = [float(file['fun']) for file in self.opts.chi2]
	    self.dm   = [float(file['juno']['pmns']['DeltaMSqEE']['value']) for file in self.opts.dm]

    def plot(self):
	fig = plt.figure()
	ax = plt.subplot(111, xlabel=r'$\Delta m^2_\mathrm{ee}$', ylabel=r'$\Delta \chi^2$', title='Sensivitivy versus mass splitting')
	ax.minorticks_on()
	ax.grid()

	ax.plot(self.dm, self.chi2, '-', markerfacecolor='none')

	ax.axvline(2.43e-3, label='YB', linestyle='--', color='red')
	ax.axvline(2.5e-3, label='Current', linestyle='--', color='green')

	ax.legend()

	f = ax.xaxis.get_major_formatter()
	f.set_powerlimits((-2,2))
	f.useMathText=True

	savefig(self.opts.output)

	plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace
    parser = ArgumentParser(description=__doc__)
    def loader(name):
        with open(name, 'r') as input:
            return load(input, Loader)

    parser.add_argument('--chi2', nargs='+', help='Yaml file to load', type=loader)
    parser.add_argument('--dm', nargs='+', help='Yaml file to load', type=loader)
    parser.add_argument('--data', nargs=2, type=np.loadtxt)
    parser.add_argument('-o', '--output')

    plotter=Plotter(parser.parse_args())
    plotter.plot()
