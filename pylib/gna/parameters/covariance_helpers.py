import ROOT
from gna.env import env
import numpy as np
import itertools

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
    
def covariate_pars(pars, in_cov_matrix):
    if isinstance(in_cov_matrix, np.ndarray):
        cov_matrix = in_cov_matrix
    else:
        cov_matrix = np.array(in_cov_matrix)

    is_covmat(pars, cov_matrix)

    for first, second in itertools.combinations_with_replacement(range(len(pars)), 2):
        pars[first].setCovariance(pars[second], cov_matrix[first, second])

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

