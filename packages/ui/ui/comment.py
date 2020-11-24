"""Commenting UI. All the arguments are ignored and needed only for annotation."""

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

    __tldr__ = {
            "" : """\
                This module may be used to insert comments into the commandline.

                \033[32mThe command will print the arguments upon execution and does nothing more.
                \033[31m./gna \\
                        -- comment Initialize a gaussian peak with default configuration and 50 bins \\
                        -- gaussianpeak --name peak_MC --nbins 50\033[0m
            """,
            }
