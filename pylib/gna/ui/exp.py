from __future__ import print_function
from gna.ui import basecmd
import argparse
import os.path
from pkgutil import iter_modules
from gna.config import cfg
import sys

expmodules = {name: loader for loader, name, _ in iter_modules(cfg.experimentpaths)}

class cmd(basecmd):
    parserkwargs=dict(add_help=False)
    @classmethod
    def initparser(cls, parser, env):
        group = parser.add_mutually_exclusive_group()
        group.add_argument('experiment', nargs='?', choices=expmodules.keys(), metavar='exp', help='experiment to load')
        group.add_argument('-e', '--exp', nargs='*', default=(), help='experiment to load')
        group.add_argument('-L', '--list-experiments', action='store_true', help='list available experiments')
        parser.add_argument('expargs', nargs=argparse.REMAINDER, help='arguments to pass to the experiment')
        parser.add_argument('--ns', help='namespace')
        parser.add_argument('-h', '--help', action='store_true', help='print help')
        cls.parser=parser

    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

        self.expname = self.opts.experiment or '_'.join(self.opts.exp)

        if self.opts.help:
            self.parser.print_help()
            print()

        if not self.expname or self.opts.list_experiments:
            print("Search paths: ", ', '.join(cfg.experimentpaths))
            print("UI exp list of experiments:")
            map(lambda l: print('   ', l), expmodules.keys())

            sys.exit(0)

    def init(self):
        expmodule = expmodules[self.expname].find_module(self.expname).load_module(self.expname)
        expcls = getattr(expmodule, 'exp')

        ns = self.env.globalns(self.opts.ns)

        parser = argparse.ArgumentParser(prog='gna -- exp '+self.expname)
        expcls.initparser(parser, ns)
        if self.opts.help:
            print('Experiment %s help'%self.expname)
            parser.print_help()
            sys.exit(0)

        expopts = parser.parse_args(self.opts.expargs)
        with ns:
            self.exp_instance = expcls(ns, expopts)
