from gna.ui import basecmd
from importlib import import_module
class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--push', nargs='+', default=[],
                            metavar='NS',
                            help='push namespaces NS to current view')
        parser.add_argument('--pop', nargs='+', default=[],
                            metavar='NS',
                            help='pop namespaces NS out of current view')
        parser.add_argument('--route', nargs=2, action='append',
                            default=[])
        parser.add_argument('--loadset', action='append', nargs=2,
                            metavar=('NS', 'PARS'),
                            default=[])
        parser.add_argument('--define', action='append', nargs='+',
                            metavar=('PAR ARG', 'ARG'),
                            default=[])

        parser.add_argument('--sigma', action='append', nargs=2,
                            metavar=('PAR', 'SIGMA'),
                            default=[])
        parser.add_argument('--central', action='append', nargs=2,
                            metavar=('PAR', 'CENTRAL'),
                            default=[])
        parser.add_argument('--value', action='append', nargs=2,
                            metavar=('PAR', 'VALUE'),
                            default=[])

        parser.add_argument('--correlation', action='append', nargs=3,
                            metavar=('PAR1', 'PAR2', 'CORR'),
                            default=[])

    def init(self):
        self.env.nsview.add([self.env.ns(x) for x in self.opts.push])
        self.env.nsview.remove([self.env.ns(x) for x in self.opts.pop])

        for ns1, ns2 in self.opts.route:
            self.env.ns(ns1).rules.append((None, ns2))

        for nsname, parsetname in self.opts.loadset:
            mod = import_module("gna.parameters.{0}".format(parsetname))
            mod.defparameters(self.env.ns(nsname))

        for define in self.opts.define:
            name, kwargs = define[0], define[1:]
            kwargs = dict(kw.split('=', 1) for kw in kwargs)
            self.env.defparameter(name, **kwargs)

        for name, sigma in self.opts.sigma:
            p = self.env.parameters[name]
            p.setSigma(p.cast(sigma))

        for name, central in self.opts.central:
            p = self.env.parameters[name]
            p.setCentral(p.cast(central))

        for name, value in self.opts.value:
            p = self.env.parameters[name]
            p.set(p.cast(value))

        for name1, name2, corr in self.opts.correlation:
            self.env.parameters[name1].setCorrelation(self.env.parameters[name2], float(corr))
