"""Parameters v02 bundle

Implements a set of parameters. It is extended implementation of the Parameters v01 bundle.
New features include:
    - Correlations may be loaded from the file
"""

from load import ROOT as R
from gna.bundle.bundle import *
import numpy as N
from gna import constructors as C

class parameters_v02(TransformationBundle):
    def __init__(self, *args, **kwargs):
        self._par_container = []
        TransformationBundle.__init__(self, *args, **kwargs)

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

        self._load_correlation_matrix()

    def build(self):
        pass

    def _load_correlation_matrix(self):
        filename = self.cfg.get('correlations', None)
        if filename is None:
            return

        mat = N.loadtxt(filename)

        if mat.shape[0]!=mat.shape[1]:
            raise Exception('Non square matrix provided:', mat.shape[0], mat.shape[1])

        if len(self._par_container)!=mat.shape[0]:
            raise Exception('Unable to set correlation to %i parameters with %ix%i matrix'%(len(pars, mat.shape[0], mat.shape[1])))

        mmin, mmax = mat.min(), mat.max()
        if mmin<-1-1.e-12 or mmax>1.0+1.e-12:
            raise Exception('Invalid min/max correlation values:', mmin, mmax)

        diag = mat.diagonal()
        ones = diag==1.0
        if not ones.all():
            raise Exception('There are values !=1 on the diagonal (d-1): ', diag[~ones]-1.0)

        if self.cfg.get('verbose', 0)>1:
            print('Load correlation matrix from %s:'%filename)
            print(mat)

        from gna.parameters import covariance_helpers as ch
        ch.covariate_pars(self._par_container, mat)
