from gna.ui import basecmd, append_typed, at_least
import ROOT
import numpy as np
from gna.env import env
from itertools import chain
from gna.dataset import Dataset

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--name', required=True)
        parser.add_argument('--pull', action='append')
        parser.add_argument('--asimov-data', nargs=2, action='append',
                            metavar=('THEORY', 'DATA'),
                            default=[])

    def run(self):
        dataset = Dataset(desc=None)
        for pullpath in self.opts.pull:
            par = env.pars[pullpath]
            dataset.assign(par, [par.central()], [par.sigma()**2])
            print (par, [par.central()], [par.sigma()**2])
        for theory_path, data_path in self.opts.asimov_data:
            dataset.assign(env.get(theory_path),
                           env.get(data_path),
                           env.get(data_path))
        self.env.parts.dataset[self.opts.name] = dataset
