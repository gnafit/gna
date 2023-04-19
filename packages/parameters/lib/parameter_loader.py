from gna.env import env
from gna.config import cfg
import ROOT
import numpy as np
import fnmatch as fn

ParameterDouble = ROOT.Parameter("double")
def __is_independent(par):
    return isinstance(par, ParameterDouble)

__special_chars = list('*?[]!')

def get_parameters(params, *, drop_fixed=True, drop_free=True, drop_constrained=False,
                   namespace=env.globalns, recursive=True):
    pargroups = []
    for candidate in params:
        cpars = []
        pargroups.append(cpars)
        if __is_independent(candidate):
            cpars.append(candidate)
            continue
        if any(char in candidate for char in __special_chars):
            matched_names = fn.filter((_[0] for _ in env.globalns.walknames()), candidate)
            matched_pars = map(env.get, matched_names)
            cpars.extend([p for p in matched_pars if __is_independent(p)])
            continue
        try:
            par = namespace[candidate]
            if __is_independent(par):
                cpars.append(par)
        except KeyError:
            par_namespace = namespace(candidate)
            if par_namespace is not env.globalns:
                independent_pars  = [par for _, par in par_namespace.walknames()
                                     if __is_independent(par)]
                cpars.extend(independent_pars)
            else:
                par = env.globalns.get(candidate)
                if __is_independent(par):
                    cpars.append(candidate)

    pars = []
    for candidate, pargroup in zip(params, pargroups):
        if not pargroup:
            raise Exception(f'Getting {candidate!s} yielded no parameters!')

        pars.extend(pargroup)

    if drop_fixed:
        pars = [par for par in pars if not par.isFixed()]

    if drop_free:
        pars = [par for par in pars if not par.isFree()]

    if drop_constrained:
        pars = [par for par in pars if par.isFree() or par.isFixed()]

    return pars

