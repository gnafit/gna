# encoding: utf-8

"""Set GNA CWD"""

from __future__ import print_function
from gna.ui import basecmd
from packages.env.lib.cwd import set_cwd, set_prefix, get_processed_paths

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('cwd', nargs='?', help='CWD to set')
        parser.add_argument('-p', '--prefix', help='Prefix')
        parser.add_argument('-d', '--print', '--dump', action='store_true', help='Print all the processed paths')

    def init(self):
        if self.opts.cwd:
            set_cwd(self.opts.cwd)
            print('CWD:', self.opts.cwd)

        if self.opts.prefix:
            set_prefix(self.opts.prefix)
            print('Prefix:', self.opts.prefix)

    def run(self):
        if self.opts.print:
            print('List of saved files:')
            map(lambda s: print('    -', s), get_processed_paths())


