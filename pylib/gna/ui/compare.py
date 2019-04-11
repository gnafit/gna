from __future__ import print_function
from gna.ui import basecmd
import argparse
import os.path
from pkgutil import iter_modules
from gna.config import cfg
import runpy
import sys
from gna.env import env


class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('experiments', nargs=2, help='Models to compare')
        parser.add_argument('--mapping', default=None, help='Mappings between names of observables')
        parser.add_argument('--ns', help='namespace')
        parser.add_argument('--output', help='Path where to save pics')

    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)
        self.observables = dict() 
        if self.opts.mapping:
            self.load_mapping()

    def init(self):
        self.check_exps_correct()
        self.extract_observables()

    def extract_observables(self):
        idx_pairs = self.mapping['indices']
        observables = self.mapping['observables']
        for obs_1, obs_2 in observables:
            for idx_1, idx_2 in idx_pairs:
                first_obs = obs_1.format(idx_1)
                second_obs = obs_2.format(idx_2)
                exp_1, exp_2 = self.exps
                print("Extracting {}".format(first_obs))
                env.get('{0}/{1}'.format(exp_1, first_obs))
                print("Extracting {}".format(second_obs))
                env.get('{0}/{1}'.format(exp_2, second_obs))

    def load_mapping(self):
        map_path = os.path.abspath(self.opts.mapping)
        loaded = runpy.run_path(map_path)
        self.exps, self.mapping = loaded['exps'], loaded['mapping']

    def check_exps_correct(self):
        if self.exps != tuple(self.opts.experiments):
                reverse = tuple(reversed(self.opts.experiments))
                if reverse != self.exps:
                    raise Exception("Experiments from cli {0} and mapping "
                                    "config  {1} doesn't match".format(self.opts.experiments, self.exps))
                self.exps = reverse
