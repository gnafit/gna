"""
Saves the command line to a file.

The command then may be repeated and should produce the same output.
"""

from gna.ui import basecmd
import pipes
from env.lib.cwd import update_namespace_cwd, get_path
from sys import argv

class cmd_save(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('output', nargs='*', help='filename to save cmd')
        parser.add_argument('-v', '--verbose', action='store_true', help='print the command line')

        redirect = parser.add_mutually_exclusive_group()
        redirect.add_argument('-r', '--redirect', action='store_true', help='add redirection')
        redirect.add_argument('-t', '--tee', action='store_true', help='add redirection via tee')

    def init(self):
        update_namespace_cwd(self.opts, 'output')
        self.level  = 0
        cmd = argv[0]
        opts = list(argv[1:])

        self.out = cmd

        self.inc()
        for opt in opts:
            self.append(opt)

        if self.opts.redirect or self.opts.tee:
            outname = get_path('stdout.out')
            errname = get_path('stderr.out')

            if self.opts.redirect:
                self.out+=(f'\\\n2>{errname} >{outname}')
            else:
                self.out+=(f'\\\n2>{errname} | tee {outname}')

        if self.opts.verbose:
            print('Command line:')
            print(self.out)

        header = '#!/bin/bash\n\n'
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

    @classmethod
    def call(cls, output: str, verbose: bool=True, redirect: bool=True, tee: bool=True):
        instance = cls(env=None, opts={'output': (output,), 'verbose': verbose, 'redirect': redirect, 'tee': tee})
        instance.init()

    __tldr__ = """\
               The main argument is the output file name to save the command.

               Save the whole command to the file 'command.sh':
               ```sh
               ./gna \\
                   -- comment Initialize a gaussian peak with default configuration and 50 bins \\
                   -- gaussianpeak --name peak_MC --nbins 50 \\
                   -- cmd-save command.sh
               ```

               In the verbose mode it also prints the command to the stdout.
               """

