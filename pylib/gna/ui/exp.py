from gna.ui import basecmd
import argparse
import gna.exp
import os.path
from pkgutil import iter_modules

path = os.path.dirname(gna.exp.__file__)
expmodules = {name: loader for loader, name, _ in iter_modules([path])}

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('experiment', choices=expmodules.keys())
        parser.add_argument('expargs', nargs=argparse.REMAINDER)

    def init(self):
        expname = self.opts.experiment
        expmodule = expmodules[expname].find_module(expname).load_module(expname)
        expcls = getattr(expmodule, 'exp')

        parser = argparse.ArgumentParser()
        expcls.initparser(parser)
        expopts = parser.parse_args(self.opts.expargs)

        expcls(self.env, expopts)
