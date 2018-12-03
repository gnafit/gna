from __future__ import print_function
from gna.env import env
from gna.config import cfg
import ROOT

def __is_independent(par):
    return isinstance(par, ROOT.Parameter("double"))

def get_parameters(params, drop_fixed=True, drop_free=True):
    pars = []
    for candidate in params:
        if isinstance(candidate, ROOT.Parameter('double')):
            pars.append(candidate)
            continue
        try:
            par_namespace = env.ns(candidate)
            par_namespace.walknames().next()
            independent_pars  = [par for _, par in par_namespace.walknames()
                                 if __is_independent(par)]
            pars.extend(independent_pars)
        except StopIteration:
            if cfg.debug_par_fetching:
                print("{0} is not a namespace, trying to use it as a parameter".format(candidate))
            pars.append(env.pars[candidate])

    if drop_fixed:
        pars = [par for par in pars if not par.isFixed()]

    if drop_free:
        pars = [par for par in pars if not par.isFree()]

