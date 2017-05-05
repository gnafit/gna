import ROOT
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
        parser.add_argument('--par_namespace', nargs='*')

    def init(self):
        minimizer = minimizers[self.opts.type](self.opts.statistic)

        parameters = [self.env.pars[pname] for pname in self.opts.par]

        # iterate over namespaces passed and extract parameters for minimization
        # TODO: Refactor to handle nested namespaces
        if self.opts.par_namespace:
            par_from_ns = []
            for par_ns in self.opts.par_namespace:
                for key in self.env.ns(par_ns).iterkeys():
                    par_name = par_ns + '.' + key
                    param = self.env.pars[par_name]
                    if isinstance(param, ROOT.GaussianParameter("double")):
                        par_from_ns.append(param)

            parameters.extend(par for par in par_from_ns if par not in parameters)
            for _ in parameters:
                print _.name()


        minimizer.addpars(parameters)

        if self.opts.spec is not None:
            minimizer.spec = spec.parse(self.env, minimizer, self.opts.spec)

        self.env.parts.minimizer[self.opts.name] = minimizer
