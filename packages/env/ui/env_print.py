"""Print given path within env"""

from __future__ import print_function
from gna.ui import basecmd
from tools.dictwrapper import DictWrapper, DictWrapperVisitor
import yaml

class DictWrapperPrinter(DictWrapperVisitor):
    fmt = u'{depth!s:>{depthlen}.{depthlen}s} {key!s:<{keylen}.{keylen}s} {vtype!s:<{typelen}.{typelen}s} {value}'
    opts = dict(keylen=30, typelen=15, depthlen=5)
    header = dict(depth='Depth', key='Key', vtype='Type', value='Value')
    def __init__(self, title, valuelen=None, keylen=None):
        self._title = title

        if valuelen is not None:
            self.fmt = self.fmt.replace('{value}', '{value!s:.{valuelen}}')
            self.opts['valuelen'] = valuelen
            self.header['value'] = 'Value ({})'.format(valuelen)

        if valuelen is not None:
            self.opts['keylen'] = valuelen

    def typestring(self, v):
        if isinstance(v, DictWrapper):
            v=v.unwrap()
        return type(v).__name__

    def start(self, d):
        print('Printing env:', self._title)
        print(self.fmt.format(**dict(self.header, **self.opts)))

    def stop(self, d):
        print()

    def enterdict(self, k, d):
        if not k:
            return
        key = '.'.join(k)
        print(self.fmt.format(depth=len(k), key=key, vtype=self.typestring(d), value='<print below>', **self.opts))

    def exitdict(self, k, d):
        pass

    def visit(self, k, v):
        depth = len(k)
        key = '.'.join(k)
        value = str(v)
        valuelen = self.opts.get('valuelen')
        if valuelen is not None:
            if len(value)>valuelen-1:
                value = value[:valuelen-1]+u'…'
        print(self.fmt.format(depth=depth, key=key, vtype=self.typestring(v), value=value, **self.opts))

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='*', default=((),), help='paths to print')
        parser.add_argument('-l', '--valuelen', type=int, help='value length')
        parser.add_argument('-k', '--keylen', type=int, help='key length')
        # parser.add_argument('-v', '--verbose', action='count', help='be more verbose')

    def init(self):
        storage = self.env.future
        for path in self.opts.paths:
            printer = DictWrapperPrinter(path, valuelen=self.opts.valuelen, keylen=self.opts.keylen)
            try:
                data = storage[path]
            except KeyError:
                raise Exception('Unable to read data path: {}'.format(path))
            data.visit(printer)
