# encoding: utf-8

"""Global environment configuration"""

from __future__ import print_function
from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('output', nargs='?', help='filename to save cmd')
        parser.add_argument('-v', '--verbose', action='store_true', help='print the command line')

    def init(self):
        from sys import argv
        self.level  = 0
        cmd = argv[0]
        opts = argv[1:]

        self.out = cmd

        self.inc()
        for opt in opts:
            self.append(opt)

        if self.opts.verbose:
            print('Command line:')
            print(self.out)

        header = '#!/usr/bin/bash\n\n'
        if self.opts.output:
            with open(self.opts.output, 'w') as f:
                f.writelines([header, self.out, ''])
            print('Command line saved to:', self.opts.output)

    def newline(self):
        self.out+=' \\\n'+self.level*'    '

    def inc(self):
        self.level+=1

    def append(self, opt):
        if opt=='--':
            self.newline()

        self.out+=opt+' '
