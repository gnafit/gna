"""Simple fit module. Makes the minimizer fit the model."""

from __future__ import print_function
from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper, DictWrapperPrinter
import yaml

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='+', help='paths to print')
        # parser.add_argument('-v', '--verbose', action='count', help='be more verbose')

    def init(self):
        printer = DictWrapperPrinter()
        for path in self.opts.paths:
            try:
                data = self.env.future[path]
                data.visit(printer)
            except KeyError:
                raise Exception('Unable to read data path: '+path)
