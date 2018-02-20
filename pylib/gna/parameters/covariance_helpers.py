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
    return cov_matrix * tmp

def __keep_absolute(**kwargs):
    pass

def __override_absolute(**kwargs):
    pass

def __override_relative(**kwargs):
    cov_mat = kwargs.pop('cov_mat')
    pars = kwargs.pop('pars')
    _uncorr_uncs = kwargs.pop('uncorr_from_file')
    tmp = np.vstack([_uncorr_uncs for _ in range(len(_uncorr_uncs))]) * _uncorr_uncs[..., np.newaxis]  
    return cov_matrix * tmp

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
        self.passed_pars = [pars]

    def covariate_pars(self):
        policy = self.cov_store['policy']
        mode = self.cov_store['mode']
        uncorr_uncs = self.cov_store['uncorr_uncs']
        pars_to_covariate = [par for par in get_parameters(self.passed_pars) 
                             if par.name() in self.cov_store['params']]
        if all(_.name() in self.cov_store['params'] for _ in pars_to_covariate):
            # put params in order to match order in covariance matrix
            pars_to_covariate.sort(key=lambda x: self.cov_store['params'].index(x.name()))
            covariate_pars(pars=pars_to_covariate,
                           cov_mat = self.cov_store['cov_mat'],
                           mode=mode,
                           policy=policy,
                           uncorr_uncs=uncorr_uncs)
 


class CovarianceStorage(object):
    """Simple class to store the order of parameters and corresponding
    covariance matrix. Meant to be used with covariate_ns() function."""
    def __init__(self, name, pars, cov_matrix):
        self.pars = pars
        self.cov_matrix = cov_matrix
        is_covmat(self.pars, self.cov_matrix)
        self.name = name

    def get_pars(self):
        return self.pars

    def get_cov(self, par1, par2):
        try:
            idx1, idx2 = self.pars.index(par1), self.pars.index(par2)
        except ValueError:
            raise ValueError("Some of pars {0} and {1} is not in covariance "
                             "storage".format(par1, par2))
        return self.cov_matrix[idx1, idx2]
        
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
        cov_matrix = cov_mat
    else:
        cov_matrix = np.array(cov_mat)
    
    keyword_args = {'cov_mat': cov_matrix,
                    'uncorr_from_file': uncorr_uncs,
                    'pars': pars,}

                    

    if (mode == 'relative') and (policy == 'keep'):
        _uncorr_uncs = np.array([par.sigma() for par in pars])
    elif (policy == 'override':
        _uncorr_uncs = np.array(uncorr_uncs)

    print(cov_matrix)
    tmp = np.vstack([_uncorr_uncs for _ in range(len(_uncorr_uncs))]) * _uncorr_uncs[..., np.newaxis]  
    print(tmp)
    _cov_mat = cov_matrix
    cov_matrix = cov_matrix * tmp
    print(cov_matrix)

    is_covmat(pars, cov_matrix)

    for first, second in itertools.combinations_with_replacement(range(len(pars)), 2):
        sigma_1, sigma_2 = _uncorr_uncs[first], _uncorr_uncs[second]
        covariance = cov_matrix[first, second]
        if first == second:
            print('In mode = {mode} and with policy = {policy}\n'
                  'sigma_1 = sigma_2 = {0}, '
                  'uncorrelated unc = {0} = {1} '
                  .format(sigma_1, np.sqrt(covariance),
                          mode=mode, policy=policy))
        else:
            print('In mode = {mode} and with policy = {policy}\n'
                  'sigma_1 = {0}, sigma_2 = {1}, '
                  'covariance = {0} * {1} * {2} = {3}'
                  .format(sigma_1, sigma_2, _cov_mat[first, second],
                          covariance, mode=mode, policy=policy))
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

