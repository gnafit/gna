from gna.ui import basecmd, set_typed

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('minimizer', action=set_typed(env.parts.minimizer))

    def init(self):
        print self.opts.minimizer.fit()

