from __future__ import print_function
from gna.ui import basecmd, set_typed
from gna.configurator import NestedDict

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('minimizer', action=set_typed(env.parts.minimizer), help='Minimizer to use', metavar='NAME')
        parser.add_argument('-o', '--output', help='Output file (hdf5)', metavar='FILENAME')

    def init(self):
        res=self.opts.minimizer.fit()

        print('Fit result:', end='')
        print(NestedDict(res.__dict__))

