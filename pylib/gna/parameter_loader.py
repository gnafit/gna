from __future__ import print_function
from gna.env import env
from gna.config import cfg
import ROOT

def is_independent(par):
    return isinstance(par, ROOT.Parameter("double"))

def get_parameters(params):
    pars = []
    for candidate in params:
        try:
            pars.append(env.pars[candidate])
        except KeyError:
            if cfg.debug_par_fetching: 
                print("Parameter {0} not found, trying to use it as a namespace".format(candidate))
            par_namespace = env.ns(candidate)
            #check that there is something in namespace
            try:
                par_namespace.walknames().next()
            except StopIteration:
                raise KeyError("Parameter {} can't be found or used as namespace".format(candidate))
            independent_pars  = [par for _, par in par_namespace.walknames()
                                 if is_independent(par)]
            pars.extend(independent_pars)
    return pars

            


            


