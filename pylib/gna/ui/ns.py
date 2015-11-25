from gna.ui import basecmd

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--push', nargs='+', default=[],
                            metavar='NS',
                            help='push namespaces NS to curent view')
        parser.add_argument('--pop', nargs='+', default=[],
                            metavar='NS',
                            help='push namespaces NS to curent view')

    def init(self):
        self.env.nsview.add([self.env.ns(x) for x in self.opts.push])
        self.env.nsview.remove([self.env.ns(x) for x in self.opts.pop])
