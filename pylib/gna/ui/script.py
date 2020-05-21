# -*- coding: utf-8 -*-
from gna.ui import basecmd
import argparse
import numpy as np
import ROOT

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        super(cmd, cls).initparser(parser, env)
        parser.add_argument('script', type=file)
        parser.add_argument("args", nargs=argparse.REMAINDER)

    def run(self):
        exec(self.opts.script.read())
