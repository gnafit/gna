from __future__ import print_function
from gna.env import env
from gna.config import cfg
import ROOT

def is_independent(par):
    return isinstance(par, ROOT.Parameter("double"))

def get_parameters(params, drop_fixed=True):
    pars = []
    for candidate in params:
        try:
            par_namespace = env.ns(candidate)
            independent_pars  = [par for _, par in par_namespace.walknames()
                                 if is_independent(par)]
            pars.extend(independent_pars)
        except AttributeError:
            if cfg.debug_par_fetching:
                print("{0} is not a namespace, trying to use it as a parameter".format(candidate))
            pars.append(env.pars[candidate])
    if drop_fixed:
        return [par for par in pars if not par.isFixed()]
    else:
        return pars
