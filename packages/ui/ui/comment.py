"""Commenting UI. Arguments are ignored."""

from gna.ui import basecmd
from argparse import REMAINDER
import pipes

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('args', nargs=REMAINDER, help="ignored arguments. First argument should not start with '-'")

    def init(self):
        print()
        print('Comment:', end='', sep='')
        s= ' '.join(pipes.quote(arg) for arg in self.opts.args)
        nlines = s.count('\n')
        if nlines>1:
            print()
        else:
            print(end=' ')

        print(s)
