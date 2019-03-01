"""Analysis module combines multiple datasets for the analysis (fit)"""
from gna.ui import basecmd, append_typed, at_least
import ROOT
import numpy as np
from itertools import chain
from gna.dataset import Dataset
from gna.parameters.parameter_loader import get_parameters

class cmd(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

        if self.opts.observables:
            if len(self.opts.observables)!=len(self.opts.datasets):
                raise Exception("Number of observables should be the same as number of datasets or 0")

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-d', '--datasets', nargs='+', required=True,
                            type=env.parts.dataset,
                            metavar='dataset', help='datasets to use')
        parser.add_argument('-p', '--parameters', nargs='+', default=[],
                            metavar='par',
                            help='parameters for the covariance matrix')
        parser.add_argument('-n', '--name', required=True, help='analysis name', metavar='name')
        parser.add_argument('-o', '--observables', nargs='+',
                            metavar='observable', help='observables (model) to be fitted')
        parser.add_argument('--toymc', choices=['covariance', 'poisson'], help='use random sampling to variate the data')

    def run(self):
        dataset = Dataset(bases=self.opts.datasets)
        parameters = get_parameters(self.opts.parameters)
        if self.opts.observables:
            observables = map(self.env.get, self.opts.observables)
        else:
            observables = None

        blocks = dataset.makeblocks(observables, parameters)

        if self.opts.toymc:
            if self.opts.toymc == 'covariance':
                toymc = ROOT.CovarianceToyMC()
            elif self.opts.toymc == 'poisson':
                toymc = ROOT.PoissonToyMC()
            for block in blocks:
                toymc.add(block.theory, block.cov)
            blocks = [block._replace(data=toymc_out)
                      for (block, toymc_out) in zip(blocks, toymc.toymc.outputs.itervalues())]
            self.env.parts.toymc[self.opts.name] = toymc
        self.env.parts.analysis[self.opts.name] = blocks
