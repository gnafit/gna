# encoding: utf-8

"""Set GNA CWD"""

from __future__ import print_function
from gna.ui import basecmd
from packages.env.lib.cwd import set_cwd

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('cwd', help='CWD to set')

    def init(self):
	set_cwd(self.opts.cwd)
	print('CWD:', self.opts.cwd)
