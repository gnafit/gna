# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from gna.ui import basecmd, append_typed, at_least
import ROOT
import numpy as np
from itertools import chain
from gna.dataset import Dataset

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-d', '--datasets', nargs='+', required=True,
                            metavar='DATASET')
        parser.add_argument('-p', '--parameters', nargs='+', default=[],
                            metavar='PAR',
                            help='paremeters for covmatrix')
        parser.add_argument('-n', '--name', required=True)
        parser.add_argument('-o', '--observables', nargs='+', required=True,
                            metavar='OBSERVABLE')

    def run(self):
        datasets = [self.env.parts.dataset[dsname]
                    for dsname in self.opts.datasets]
        dataset = Dataset(name=self.opts.name+'_dataset',
                          desc=None,
                          bases=datasets)
        parameters = [self.env.pars[pname] for pname in self.opts.parameters]
        observables = []
        for path in self.opts.observables:
            nspath, name = path.split('/')
            observables.append(self.env.ns(nspath).observables[name])

        blocks = dataset.makeblocks(observables, parameters)
        self.env.parts.prediction[name] = blocks
        print(blocks)
