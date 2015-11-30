from gna.ui import basecmd
from importlib import import_module
class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--push', nargs='+', default=[],
                            metavar='NS',
                            help='push namespaces NS to curent view')
        parser.add_argument('--pop', nargs='+', default=[],
                            metavar='NS',
                            help='push namespaces NS to curent view')
        parser.add_argument('--route', nargs=2, action='append',
                            default=[])
        parser.add_argument('--loadset', action='append', nargs=2,
                            metavar=('NS', 'PARS'),
                            default=[])
        parser.add_argument('--define', action='append', nargs='+',
                            metavar=('NAME ARG', 'ARG'),
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
            if '.' in name:
                nsname, name = name.rsplit('.', 1)
                self.env.ns(nsname).defparameter(name, **kwargs)
            else:
                self.env.globalns.defparameter(name, **kwargs)
