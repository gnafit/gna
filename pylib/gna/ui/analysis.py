from gna.ui import basecmd, append_typed, at_least
import ROOT
import numpy as np
from itertools import chain
from gna.dataset import Dataset
from gna.parameters.parameter_loader import get_parameters

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-d', '--datasets', nargs='+', required=True,
                            type=env.parts.dataset,
                            metavar='DATASET')
        parser.add_argument('-p', '--parameters', nargs='+', default=[],
                            metavar='PAR',
                            help='parameters for covariance matrix')
        parser.add_argument('-n', '--name', required=True)
        parser.add_argument('-o', '--observables', nargs='+', required=True,
                            metavar='OBSERVABLE')
        parser.add_argument('--toymc', choices=['covariance', 'poisson'])

    def run(self):
        dataset = Dataset(bases=self.opts.datasets)
        parameters = get_parameters(self.opts.parameters)
        observables = [self.env.get(path) for path in self.opts.observables]

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
