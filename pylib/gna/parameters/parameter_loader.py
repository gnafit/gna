from __future__ import print_function
from gna.env import env
from gna.config import cfg
import ROOT

ParameterDouble = ROOT.Parameter("double")
def __is_independent(par):
    return isinstance(par, ParameterDouble)

def get_parameters(params, drop_fixed=True, drop_free=True):
    special_chars = list('*?[]!')
    pars = []
    for candidate in params:
        if __is_independent(candidate):
            pars.append(candidate)
            continue
        if any(char in candidate for char in special_chars):
            import fnmatch as fn
            matched_names = fn.filter((_[0] for _ in env.globalns.walknames()), candidate)
            matched_pars = map(env.get, matched_names)
            pars.extend(matched_pars)
            continue
        try:
            par = env.pars[candidate]
            pars.append(par)
        except KeyError:
            par_namespace = env.ns(candidate)
            if par_namespace != env.globalns:
                independent_pars  = [par for _, par in par_namespace.walknames()
                                     if __is_independent(par)]
            else:
                independent_pars = [env.globalns.get(candidate)]
            pars.extend(independent_pars)

    if drop_fixed:
        pars = [par for par in pars if not par.isFixed()]

    if drop_free:
        pars = [par for par in pars if not par.isFree()]

    return pars

