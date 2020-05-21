# -*- coding: utf-8 -*-
"""Plot 1d ovservables"""

from __future__ import absolute_import
from __future__ import print_function
from gna.ui import basecmd, append_typed, qualified
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_tools.helpers import savefig
import numpy as np
from gna.bindings import common
from gna.env import PartNotFoundError, env

class cmd(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    @classmethod
    def initparser(cls, parser, env):
        def observable(path):
            try:
                return env.ns('').getobservable(path)
            except KeyError:
                raise PartNotFoundError("observable", path)

        parser.add_argument('data', metavar=('DATA',), type=observable, help='observable to store')
        parser.add_argument('output', help='filename')
        parser.add_argument('--header', default='', help='Header')

    def run(self):
        data = self.opts.data.data()
        dt   = self.opts.data.datatype()

        header = self.opts.header
        if dt.isHist():
            if dt.shape.size()!=1:
                raise Exception('2d histograms not yet implemented')

            edges = np.array(dt.edges)
            edges_left, edges_right = edges[:-1], edges[1:]
            dump = edges_left, edges_right, data

            if not header:
                header = 'bin_left bin_right data'
        elif dt.isPoints():
            dump = data,

            if not header:
                header = 'data'
        else:
            raise Exception('DataType is undefined')

        dump = np.array(dump).T

        try:
            np.savetxt(self.opts.output, dump, header=header)
        except:
            raise Exception('Unable to write data to: '+self.opts.output)

        print(('Dump data to: '+self.opts.output))
