from gna.ui import basecmd, append_typed, at_least
import ROOT
import numpy as np
from gna.env import PartNotFoundError
from itertools import chain
from gna.dataset import Dataset

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-d', '--datasets', nargs='+', required=True,
                            type=env.parts.dataset,
                            metavar='DATASET')
        parser.add_argument('-p', '--parameters', nargs='+', default=[],
                            metavar='PAR',
                            help='paremeters for covmatrix')
        parser.add_argument('-n', '--name', required=True)
        parser.add_argument('-o', '--observables', nargs='+', required=True,
                            metavar='OBSERVABLE')

    def run(self):
        dataset = Dataset(bases=self.opts.datasets)
        parameters = [self.env.pars[pname] for pname in self.opts.parameters]
        observables = []
        for path in self.opts.observables:
            observables.append(self.env.get(path))

        blocks = dataset.makeblocks(observables, parameters)
        self.env.parts.analysis[self.opts.name] = blocks
