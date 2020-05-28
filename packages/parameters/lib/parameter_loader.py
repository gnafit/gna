from __future__ import print_function
from gna.env import env
from gna.config import cfg
import ROOT
import numpy as np

ParameterDouble = ROOT.Parameter("double")
def __is_independent(par):
    return isinstance(par, ParameterDouble)

def get_parameters(params, drop_fixed=True, drop_free=True, drop_constrained=False):
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
            if par_namespace is not env.globalns:
                independent_pars  = [par for _, par in par_namespace.walknames()
                                     if __is_independent(par)]
            else:
                independent_pars = [env.globalns.get(candidate)]

            pars.extend(independent_pars)

    if drop_fixed:
        pars = [par for par in pars if not par.isFixed()]

    if drop_free:
        pars = [par for par in pars if not par.isFree()]

    if drop_constrained:
        pars = [par for par in pars if par.isFree()]

    return pars

def get_uncertainties(parlist):
    npars = len(parlist)
    centrals, sigmas = np.zeros(npars, dtype='d'), np.zeros(npars, dtype='d')
    correlations = False

    for i, par in enumerate(parlist):
        centrals[i]=par.central()
        sigmas[i]=par.sigma()
        correlations = correlations or par.isCorrelated()

    if correlations:
        # In case there are correlations:
        # - create covariance matrix
        # - fill the diagonal it with the value of sigma**2
        # - fill the off-diagonal elements with covarainces
        # - create Points, representing the covariance matrix
        covariance = np.diag(sigmas**2)
        for i in range(npars):
            for j in range(i):
                pari, parj = parlist[i], parlist[j]
                cov = pari.getCovariance(parj)
                covariance[i,j]=covariance[j,i]=cov

        return sigmas, centrals, covariance

    return sigmas, centrals, correlations

