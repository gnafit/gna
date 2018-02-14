from gna.ui import basecmd, append_typed, at_least
import ROOT
import numpy as np
from gna.env import PartNotFoundError
from gna.parameters.parameter_loader import get_parameters
from gna.parameters.covariance_helpers import covariate_pars
from gna.config import cfg
import argparse


class cmd(basecmd):

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-a', '--assign',dest='candidates', nargs='*', action='append')

    def run(self):
        for cov, raw_pars in self.opts.candidates:
            self.__check_presense(cov)
            self.__covariate_pars(cov, [raw_pars])
            

    def __check_presense(self, cov):
        cov_available = cfg['covariances'].keys()
        if not cov in cov_available: 
            raise Exception("No such covariance set as {0}. Try something "
                            "from {1}".format(cov, cov_available))

    def __covariate_pars(self, cov, raw_pars):
        def all_names_in_storage(names_in_store, param_names):
            return all(par in param_names for par in names_in_store)

        cov_store = cfg['covariances'][cov]
        params_in_store = cov_store['params']
        pars_to_covariate = [par for par in get_parameters(raw_pars) 
                             if par.name() in params_in_store]
        if all(_.name() in params_in_store for _ in pars_to_covariate):
            # put params in order to match order in covariance matrix
            pars_to_covariate.sort(key=lambda x: params_in_store.index(x.name()))
            covariate_pars(pars_to_covariate, cov_store['cov_mat'])
        


