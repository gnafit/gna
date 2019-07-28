import ROOT
from gna.env import env
import numpy as np
import itertools
from gna.config import cfg
from parameter_loader import get_parameters

def __keep_relative(**kwargs):
    cov_mat = kwargs.pop('cov_mat')
    pars = kwargs.pop('pars')
    _uncorr_uncs = np.array([par.sigma() for par in pars])
    tmp = np.vstack([_uncorr_uncs for _ in range(len(_uncorr_uncs))]) * _uncorr_uncs[..., np.newaxis]
    return cov_mat * tmp

def __keep_absolute(**kwargs):
    raise Exception('Perhaps one shouldn\'t override uncorrelated '
                    'unceratinties in absolute mode from configuration file')


def __override_absolute(**kwargs):
    return kwargs.pop('cov_mat')

def __override_relative(**kwargs):
    cov_mat = kwargs.pop('cov_mat')
    pars = kwargs.pop('pars')
    _uncorr_uncs = kwargs.pop('uncorr_from_file')
    tmp = np.vstack([_uncorr_uncs for _ in range(len(_uncorr_uncs))]) * _uncorr_uncs[..., np.newaxis]
    return cov_mat * tmp

dispatch_modes = {('keep', 'relative'): __keep_relative,
                  ('keep', 'absolute'): __keep_absolute,
                  ('override', 'relative'): __override_relative,
                  ('override', 'absolute'): __override_absolute,}


class CovarianceHandler(object):
    def __init__(self, covariance_name, pars):
        try:
            self.covariance_obj = cfg['covariances'][covariance_name]
        except KeyError as e:
            info = 'Covariance {0} is absent in configuration. ' + 'Check the contents of {1} to see whether it is present'
            raise KeyError, info.format(covariance_name, str(cfg['covariance_path']))

        self.cov_store = cfg['covariances'][covariance_name]
        self.passed_pars = pars

    def covariate_pars(self):
        policy = self.cov_store['policy']
        mode = self.cov_store['mode']
        uncorr_uncs = np.array(self.cov_store['uncorr_uncs'])
        pars_to_covariate = [par for par in get_parameters(self.passed_pars, drop_fixed=True, drop_free=True)
                             if par.name() in self.cov_store['params']]
        if all(_.name() in self.cov_store['params'] for _ in pars_to_covariate):
            # put params in order to match order in covariance matrix
            pars_to_covariate.sort(key=lambda x: self.cov_store['params'].index(x.name()))
            covariate_pars(pars=pars_to_covariate,
                           cov_mat = self.cov_store['cov_mat'],
                           mode=mode,
                           policy=policy,
                           uncorr_uncs=uncorr_uncs)




def covariate_ns(ns, cov_storage):
    cov_ns = env.ns(ns)
    pars_in_ns = [par[1] for par in cov_ns.walknames()]
    if len(pars_in_ns) == 0:
        raise Exception("Passed namespace {} is empty".format(ns))
    pars_in_store = cov_storage.get_pars()
    mutual_pars = [par for par in pars_in_store if par in pars_in_ns]
    for par1, par2 in itertools.combinations_with_replacement(mutual_pars, 2):
        par1.setCovariance(par2, cov_storage.get_cov(par1, par2))

def covariate_pars(pars, cov_mat, mode='relative', policy='keep', uncorr_uncs=None):
    if isinstance(cov_mat, np.ndarray):
        _cov_initial = cov_mat
    else:
        _cov_initial = np.array(cov_mat)

    keyword_args = {'cov_mat': _cov_initial,
                    'uncorr_from_file': uncorr_uncs,
                    'pars': pars,}

    cov_matrix = dispatch_modes[(policy, mode)](**keyword_args)

    is_covmat(pars, cov_matrix)

    for first, second in itertools.combinations_with_replacement(range(len(pars)), 2):
        covariance = cov_matrix[first, second]
        pars[first].setCovariance(pars[second], covariance)


def is_covmat(pars, cov_matrix):
    def is_positive_semidefinite(cov_matrix):
        return np.all(np.linalg.eigvals(cov_matrix) > 0.)

    assert cov_matrix.shape[0] == cov_matrix.shape[1], (
            "Covariance matrix for parameters {} is not square".format([_.name() for _ in pars]))
    assert np.allclose(cov_matrix.transpose(), cov_matrix), (
    "Covariance matrix for parameters {} is not symmetric".format(pars))
    assert len(pars) == cov_matrix.shape[0], (
    "The number of parameters and size of covariance matrix do not match")
    assert is_positive_semidefinite(cov_matrix), (
    "The matrix is not positive-semidefinite")
    return
