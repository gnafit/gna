
"""Save the command line to a shell file

The command then may be repeated and should produce the same output
"""

from gna.ui import basecmd
import pipes
from packages.env.lib.cwd import update_namespace_cwd
from sys import argv

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('output', nargs='*', help='filename to save cmd')
        parser.add_argument('-v', '--verbose', action='store_true', help='print the command line')

    def init(self):
        update_namespace_cwd(self.opts, 'output')
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
            for opath in self.opts.output:
                with open(opath, 'w') as f:
                    f.writelines([header, self.out, '\n'])
                print('Command line saved to:', opath)

    def newline(self):
        self.out+=' \\\n'+self.level*'    '

    def inc(self):
        self.level+=1

    def append(self, opt):
        if opt=='--':
            self.newline()

        self.out+=pipes.quote(opt)+' '
