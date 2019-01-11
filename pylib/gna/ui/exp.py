from __future__ import print_function
from gna.ui import basecmd
import argparse
import os.path
from pkgutil import iter_modules
from gna.config import cfg

expmodules = {name: loader for loader, name, _ in iter_modules(cfg.experimentpaths)}

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('experiment', nargs='?', choices=expmodules.keys(), metavar='exp', help='experiment to load')
        group.add_argument('-L', '--list-experiments', action='store_true', help='list available experiments')
        parser.add_argument('expargs', nargs=argparse.REMAINDER, help='arguments to pass to the experiment')
        parser.add_argument('--ns', help='namespace')

    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

        if self.opts.list_experiments:
            print("UI exp list of experiments:")
            map(print, expmodules.keys())

            from sys import exit
            exit(0)

    def init(self):
        expname = self.opts.experiment
        expmodule = expmodules[expname].find_module(expname).load_module(expname)
        expcls = getattr(expmodule, 'exp')

        ns = self.env.globalns(self.opts.ns)

        parser = argparse.ArgumentParser()
        expcls.initparser(parser, ns)
        expopts = parser.parse_args(self.opts.expargs)

        self.exp_instance = expcls(ns, expopts)
