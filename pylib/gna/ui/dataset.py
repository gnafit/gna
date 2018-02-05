from __future__ import print_function
from gna.ui import basecmd, append_typed, at_least
import ROOT
import numpy as np
from gna.env import env
from itertools import chain
from gna.dataset import Dataset
from gna.parameters.parameter_loader import get_parameters

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--name', required=True)
        parser.add_argument('--pull', action='append')
        parser.add_argument('--asimov-data', nargs=2, action='append',
                            metavar=('THEORY', 'DATA'),
                            default=[])
        parser.add_argument('--asimov-poisson', nargs=2, action='append',
                            metavar=('THEORY', 'DATA'),
                            default=[])
        parser.add_argument('--error-type', choices=['pearson', 'neyman'],
                default='pearson', help='The type of statistical errors to be used')

    def run(self):
        dataset = Dataset(desc=None)
        if self.opts.pull:
            pull_pars = get_parameters(self.opts.pull)

            for par in pull_pars:
                dataset.assign(par, [par.central()], [par.sigma()**2])
                print (par, [par.central()], [par.sigma()**2])

        if self.opts.asimov_data:
            for theory_path, data_path in self.opts.asimov_data:
                if self.opts.error_type == 'neyman':
                    dataset.assign(env.get(theory_path),
                                   env.get(data_path),
                                   env.get(data_path))
                elif self.opts.error_type == 'pearson':
                    dataset.assign(env.get(theory_path),
                                   env.get(data_path),
                                   env.get(theory_path))

        if self.opts.asimov_poisson:
            for theory_path, data_path in self.opts.asimov_poisson:
                data_poisson = np.random.poisson(env.get(data_path).data())
                if self.opts.error_type == 'neyman':
                    dataset.assign(env.get(theory_path),
                                   data_poisson,
                                   env.get(data_path))
                elif self.opts.error_type == 'pearson':
                    dataset.assign(env.get(theory_path),
                                   data_poisson,
                                   env.get(theory_path))

        self.env.parts.dataset[self.opts.name] = dataset
