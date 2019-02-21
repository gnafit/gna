from __future__ import print_function
from gna.ui import basecmd
from importlib import import_module
from gna.config import cfg
from gna.parameters.covariance_helpers import CovarianceHandler

undefined = ['undefined']

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-n', '--name', help='the namespace to work with')
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
        parser.add_argument('--fix', action='append', nargs=1,
                            metavar=('PAR'), default=[])

        parser.add_argument('--covariance', action='append', nargs='*',
                default=[],
                            metavar=('COVARIANCE_SET', 'PARS'), help='First '
                            'argument: name of covariance matrix, rest: names '
                            'of parameters to covariate')
        #  parser.add_argument('--correlation', action='append', nargs='*',
                            #  metavar=('CORRELATION_SET', 'PARS'))

        parser.add_argument('-p', '--print', nargs='?', default=undefined, help='print namespace')

    def init(self):
        if self.opts.name:
            namespace = self.env.globalns(self.opts.name)
        else:
            namespace = self.env.globalns
        self.env.nsview.add([namespace(x) for x in self.opts.push])
        self.env.nsview.remove([namespace(x) for x in self.opts.pop])

        for ns1, ns2 in self.opts.route:
            namespace(ns1).rules.append((None, ns2))

        for nsname, parsetname in self.opts.loadset:
            mod = import_module("gna.parameters.{0}".format(parsetname))
            mod.defparameters(namespace(nsname))

        for define in self.opts.define:
            name, kwargs = define[0], define[1:]
            kwargs = dict(kw.split('=', 1) for kw in kwargs)
            namespace.defparameter(name, **kwargs)

        for name, sigma in self.opts.sigma:
            p = self.env.parameters[name]
            p.setSigma(p.cast(sigma))

        for name, central in self.opts.central:
            p = self.env.parameters[name]
            p.setCentral(p.cast(central))

        for name, value in self.opts.value:
            p = self.env.parameters[name]
            p.set(p.cast(value))

        for name in self.opts.fix:
            p = self.env.parameters[name[0]]
            p.setFixed()

        #  for name1, name2, corr in self.opts.correlation:
            #  self.env.parameters[name1].setCorrelation(self.env.parameters[name2], float(corr))

        for entry in self.opts.covariance:
            cov, pars = entry[0], entry[1:]
            CovarianceHandler(cov, pars).covariate_pars()

        try:
            if self.opts.print is not undefined:
                namespace(self.opts.print or '').printparameters(labels=True)
        except Exception as e:
            print('Unable to print namespace "%s": %s'%(self.opts.print, e.message))
