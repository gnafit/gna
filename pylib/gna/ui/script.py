from gna.ui import basecmd
import argparse
import numpy as np
import ROOT

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        super(cmd, cls).initparser(parser, env)
        parser.add_argument('script')
        parser.add_argument("args", nargs=argparse.REMAINDER)

    def run(self):
        with open(self.opts.script, 'r') as f:
            exec(f.read())

