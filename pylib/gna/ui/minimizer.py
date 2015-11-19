from gna.ui import basecmd, set_typed
from gna.minimizers import minimizers
from gna.minimizers import spec

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-s', '--spec', default=None)
        parser.add_argument('name')
        parser.add_argument('type', choices=minimizers.keys())
        parser.add_argument('statistic', action=set_typed(env.parts.statistic))
        parser.add_argument('par', nargs='*')

    def init(self):
        minimizer = minimizers[self.opts.type](self.opts.statistic)

        minimizer.addpars([self.env.pars[pname] for pname in self.opts.par])

        if self.opts.spec is not None:
            minimizer.spec = spec.parse(self.env, minimizer, self.opts.spec)

        self.env.parts.minimizer[self.opts.name] = minimizer
