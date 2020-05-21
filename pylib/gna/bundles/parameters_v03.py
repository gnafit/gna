# -*- coding: utf-8 -*-

"""Parameters v03 bundle
Based on: parameters_v02

Implements a set of parameters.
New features include:
    - [v02] Correlations may be loaded from the file
    - [v03] Covariance matrix may be loaded from the file
"""

from __future__ import print_function
from __future__ import absolute_import
from load import ROOT as R
from gna.bundle.bundle import *
import numpy as N
from gna import constructors as C

class parameters_v03(TransformationBundle):
    covmat, corrmat = None, None
    def __init__(self, *args, **kwargs):
        self._par_container = []
        TransformationBundle.__init__(self, *args, **kwargs)

        if 'correlations' in self.cfg and 'covariance' in self.cfg:
            raise Exception("Mutually exclusive options passed: correlations and covariance")

    @staticmethod
    def _provides(cfg):
        sepunc = cfg.get('separate_uncertainty', None)
        if sepunc:
            return (cfg.parameter, sepunc), ()
        else:
            return (cfg.parameter,), ()

    def define_variables(self):
        self._par_container = []
        separate_uncertainty = self.cfg.get('separate_uncertainty', False)
        parname = self.cfg.parameter
        pars = self.cfg.pars
        labelfmt = self.cfg.get('label', '')

        for it_major in self.nidx_major:
            major_values = it_major.current_values()
            if major_values:
                parcfg = pars[major_values]
            else:
                parcfg = pars

            for it_minor in self.nidx_minor:
                it=it_major+it_minor
                label = it.current_format(labelfmt) if labelfmt else ''

                if separate_uncertainty:
                    if parcfg.mode=='fixed':
                        raise self.exception('Can not separate uncertainty for fixed parameters')

                    unccfg = parcfg.get_unc()
                    uncpar = self.reqparameter(separate_uncertainty, it, cfg=unccfg, label=label+' (norm)')
                    parcfg.mode='fixed'

                par = self.reqparameter(parname, it, cfg=parcfg, label=label)

                if self.cfg.get("objectize"):
                    trans=par.transformations.value
                    trans.setLabel(label)
                    self.set_output(parname, it, trans.single())

                self._par_container.append(par)

        if 'correlations' in self.cfg:
            self._load_correlation_matrix()
        elif 'covariance' in self.cfg:
            self._load_covariance_matrix()

    def build(self):
        pass

    def _load_correlation_matrix(self):
        filename = self.cfg.get('correlations', None)
        assert filename

        corrmat = N.loadtxt(filename)
        if self.cfg.get('verbose', 0)>1:
            print('Load correlation matrix from %s:'%filename)
            print(corrmat)

        self._checkmatrix(corrmat, is_correlation=True)

        self.corrmat = corrmat

        from gna.parameters import covariance_helpers as ch
        ch.covariate_pars(self._par_container, corrmat)

    def _load_covariance_matrix(self):
        filename = self.cfg.get('covariance', None)
        assert filename

        covmat = N.loadtxt(filename)
        if self.cfg.get('verbose', 0)>1:
            print('Load covariance matrix from %s:'%filename)
            print(covmat)

        sigma_inv=N.diag(covmat.diagonal()**-0.5)
        corrmat = N.matmul(N.matmul(sigma_inv, covmat), sigma_inv)

        if self.cfg.get('verbose', 0)>1:
            print('Compute correlation matrix from:')
            print(corrmat)

        self._checkmatrix(covmat, is_correlation=False)

        self.covmat = covmat
        self.corrmat = corrmat

        for par, sigma2 in zip(self._par_container, covmat.diagonal()):
            par.setSigma(sigma2**0.5)

        from gna.parameters import covariance_helpers as ch
        ch.covariate_pars(self._par_container, corrmat)


    def _checkmatrix(self, mat, is_correlation):
        if mat.shape[0]!=mat.shape[1]:
            raise Exception('Non square matrix provided:', mat.shape[0], mat.shape[1])

        if len(self._par_container)!=mat.shape[0]:
            raise Exception('Unable to set correlation to %i parameters with %ix%i matrix'%(len(pars, mat.shape[0], mat.shape[1])))

        mmin, mmax = mat.min(), mat.max()
        if mmin<-1-1.e-12 or mmax>1.0+1.e-12:
            raise Exception('Invalid min/max correlation values:', mmin, mmax)

        diag = mat.diagonal()

        if is_correlation:
            ones = diag==1.0
            if not ones.all():
                raise Exception('There are values !=1 on the diagonal (d-1): ', diag[~ones]-1.0)
        else:
            if (diag<0).any():
                raise Exception('Covariance matrix contains values below zero')
