"""Executes a hook"""

from gna.ui import basecmd

class hook(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-r', '--root', default='hooks', metavar='name', help='root environment')
        parser.add_argument('hooks', nargs='+', metavar='path', help='hook path')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity')

    def init(self):
        storage = self.env.future
        if self.opts.root:
            storage = storage.child(self.opts.root)

        if self.opts.verbose:
            if self.opts.root:
                print()
            else:
                print('Executing hooks:')

        for hookname in self.opts.hooks:
            if self.opts.verbose:
                print(f'Executing hook [{self.opts.root}]: {hookname}')

            hook = storage[hookname]
            hook()


    __tldr__ = """\
            TBD
            """
