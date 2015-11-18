from gna.ui import basecmd
from gna.minimizers import minimizers
from gna.minimizers import spec

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser):
        parser.add_argument('-s', '--spec', default=None)
        parser.add_argument('name')
        parser.add_argument('type')
        parser.add_argument('statistic')
        parser.add_argument('par', nargs='*')

    def init(self):
        statistic = self.env.statistics[self.opts.statistic]
        minimizer = minimizers[self.opts.type](statistic)

        minimizer.addpars([self.env.pars[pname] for pname in self.opts.par])

        if self.opts.spec is not None:
            minimizer.spec = spec.parse(self.env, minimizer, self.opts.spec)

        self.env.addminimizer(self.opts.name, minimizer)
