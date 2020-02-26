# encoding: utf-8

u"""Change global parameters of the matplotlib"""

from gna.ui import basecmd
import ROOT
import numpy as np
from gna import constructors as C
import yaml
import pprint

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument( '-i', '--interactive', action='store_true', help='switch to interactive matplotlib' )
        parser.add_argument('-l', '--latex', action='store_true', help='enable latex mode')
        parser.add_argument('-r', '--rcparam', '--rc', nargs='+', default=[], help='YAML dictionary with RC configuration',
                            type=lambda s: yaml.load(s, Loader=yaml.Loader))
        parser.add_argument('-v', '--verbose', action='count', help='verbosity level')

    def init(self):
        import matplotlib as mpl
        if self.opts.interactive:
            if self.opts.verbose:
                print('Interactive matplotlib')
            mpl.pyplot.ion()

        if self.opts.latex:
            if self.opts.verbose:
                print('Matplotlib with latex')
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.unicode'] = True
            mpl.rcParams['font.size'] = 13

        if self.opts.verbose>1 and self.opts.rcparam:
            print('Matplotlib extra options')
            for d in self.opts.rcparam:
                pprint.pprint(d)

        for d in self.opts.rcparam:
            mpl.rcParams.update(d)
