"""Embed IPython repl on module execution"""

from gna.ui import basecmd
import ROOT as R
from matplotlib import pyplot as plt
import sys, scipy
import numpy as np
from textwrap import dedent

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        super().initparser(parser, env)
        parser.add_argument('code', nargs='+', default=())

    def run(self):
        for snippet in self.opts.code:
            exec(dedent(snippet))
