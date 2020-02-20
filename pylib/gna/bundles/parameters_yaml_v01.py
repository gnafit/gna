# -*- coding: utf-8 -*-

"""Parameters_yaml v01 bundle
Based on: parameters_v03

Implements a set of parameters.
New features include:
    - [v01] read data from yaml file
"""

from __future__ import print_function
from load import ROOT as R
from gna.bundle.bundle import *
import numpy as N
from gna import constructors as C
import yaml

class parameters_yaml_v01(TransformationBundle):
    covmat, corrmat = None, None
    def __init__(self, *args, **kwargs):
        self._par_container = []
        TransformationBundle.__init__(self, *args, **kwargs)

        self.check_nidx_dim(1, 1, 'major')
        self.load_data()

    def load_data(self):
        if not 'data' in self.cfg:
            raise Exception('Should provide data file')

        try:
            with open(self.cfg.data, 'r') as stream:
                data = self.data = yaml.load(stream, Loader=yaml.Loader)
        except:
            raise Exception('Unable to load input data file: '+self.cfg.data)

        for name in ('values', 'names', 'mode'):
            if not name in data:
                raise Exception('Input file {} misses field {}'.format(self.cfg.data, name))

        # if ('correlations' in data or 'uncertainties' in data) and 'covariance' in data:
            # raise Exception('Conflicting fields in data: correlations/uncertainties and covariance')

        for field in ('values', 'uncertainties', 'correlations'): #, 'covariance'
            if field in data:
                data[field] = N.asanyarray(data[field])

        if 'correlations' in data:
            if self.cfg.get('verbose', 0)>1:
                print('Correlation matrix')
                print(self.data['correlations'], )

    @staticmethod
    def _provides(cfg):
        sepunc = cfg.get('separate_uncertainty', None)
        if sepunc:
            return (cfg.parameter, sepunc), ()
        else:
            return (cfg.parameter,), ()

    def define_variables(self):
        from gna.parameters import covariance_helpers as ch

        self._par_container = []
        separate_uncertainty = self.cfg.get('separate_uncertainty', False)
        parname = self.cfg.parameter
        labelfmt = self.cfg.get('label', '')

        data = self.data
        values = data['values']
        names = data['names']
        unc = data['uncertainties']
        mode = data['mode']
        if mode=='percent':
            relunc = unc/100.
            unc = values*relunc
        elif mode=='relative':
            relunc = unc
            unc = values*relunc
        elif mode=='absolute':
            relunc = unc/values
        else:
            raise Exception('Invalid uncertainty mode: '+mode)

        for it_minor in self.nidx_minor:
            container = []

            for it_major in self.nidx_major:
                major_values, = it_major.current_values()
                if not major_values in names:
                    raise Exception('Variable {} is not in the list'.format(major_values))
                data_idx = names.index(major_values)

                it=it_major+it_minor
                label = it.current_format(labelfmt) if labelfmt else ''

                if separate_uncertainty:
                    uncpar = self.reqparameter(separate_uncertainty, it, central=1.0, sigma=relunc[data_idx],
                                               label=label+' (norm)')
                    par = self.reqparameter(parname, it, central=values[data_idx], fixed=True, label=label)
                    container.append(uncpar)
                else:
                    par = self.reqparameter(parname, it, central=values[data_idx], sigma=unc[data_idx], label=label)
                    container.append(par)

                if self.cfg.get("objectize"):
                    trans=par.transformations.value
                    trans.setLabel(label)
                    self.set_output(parname, it, trans.single())

            self._par_container.append(container)

            if 'correlations' in self.data:
                self._checkmatrix(self.data['correlations'], container, is_correlation=True)
                ch.covariate_pars(container, self.data['correlations'])
        # elif 'covariance' in self.cfg:
            # self._load_covariance_matrix()

    def build(self):
        pass

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


    def _checkmatrix(self, mat, par_container, is_correlation):
        if mat.shape[0]!=mat.shape[1]:
            raise Exception('Non square matrix provided:', mat.shape[0], mat.shape[1])

        if len(par_container)!=mat.shape[0]:
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
