"""Print given path within env"""

from __future__ import print_function
from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper, DictWrapperVisitor
import yaml

class DictWrapperPrinter(DictWrapperVisitor):
    fmt = '{depth!s:>5s} {key!s:<{keylen}s} {vtype!s:<{typelen}s} {value}'
    opts = dict(keylen=30, typelen=15)
    def __init__(self, title):
        self._title = title

    def typestring(self, v):
        if isinstance(v, DictWrapper):
            v=v.unwrap()
        return type(v).__name__

    def start(self, d):
        print('Printing env:', self._title)
        print(self.fmt.format(depth='Depth', key='Key', vtype='Type', value='Value', **self.opts))

    def stop(self, d):
        print()

    def enterdict(self, k, d):
        if not k:
            return
        print(self.fmt.format(depth=len(k), key=k[-1], vtype=self.typestring(d), value='', **self.opts))
        k = '.'.join(k)
        print('      Print {}'.format(k))

    def exitdict(self, k, d):
        pass

    def visit(self, k, v):
        depth = len(k)
        k = k[-1]
        print(self.fmt.format(depth=depth, key=k, vtype=self.typestring(v), value=v, **self.opts))

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='+', help='paths to print')
        # parser.add_argument('-v', '--verbose', action='count', help='be more verbose')

    def init(self):
        storage = DictWrapper(self.env.future, split='.')
        for path in self.opts.paths:
            printer = DictWrapperPrinter(path)
            try:
                data = storage[path]
                data.visit(printer)
            except KeyError:
                raise Exception('Unable to read data path: '+path)
