from __future__ import print_function
import ROOT
import gna.constructors as C
from collections import defaultdict, namedtuple
import numpy as np

Block = namedtuple('Block', ('theory', 'data', 'cov'))

class Dataset(object):
    def __init__(self, desc=None, bases=[]):
        self.desc = desc
        self.data = {}
        self.covariance = defaultdict(list)
        for base in reversed(bases):
            self.data.update(base.data)
        self._update_covariances(bases)

    def _update_covariances(self, bases):
        from itertools import combinations

        #Update individual covariances, i.e. unocorrelated errors
        for base in reversed(bases):
            self.covariance.update(base.covariance)

        # Check for covariations between different dataset, i.e. for presense
        #  for covariated pull terms
        underlaying_params = [(key, base.data.values()[0]) for base in bases for key in base.data.keys()
                              if isinstance(key, ROOT.Parameter('double'))
                                 and not key.isFixed() and key.isCorrelated()]
        for (par_1, base_1), (par_2, base_2) in combinations(underlaying_params, 2):
            mutual_cov =  par_1.getCovariance(par_2)
            if mutual_cov==0.0:
                continue
            self.covariance[frozenset([par_1, par_2])] = [self._pointize(mutual_cov)]

    def assign(self, obs, value, error):
        """Given observable assign value that is going to serve as data and uncertainty to it.
        """
        self.data[obs] = self._pointize(value)
        self.covariate(obs, obs, error)

    def _pointize(self, obj):
        """Given object checks whether it is PyROOT type (perhaps better to
        refactor it) and if not make a Points out of it. Use case -- turning
        numpy array into points.
        """
        if not isinstance(type(obj), ROOT.PyRootType):
            obj = C.Points(obj)
        return obj

    def covariate(self, obs1, obs2, cov):
        """ Given two observables and covariance between them, add covariate
        into a stotage for covariances
        """
        self.covariance[frozenset([obs1, obs2])].append(self._pointize(cov))

    def iscorrelated(self, obs1, obs2, covparameters):
        """Checks whether two observables are correlated or affected by covparameters
        """
        if self.covariance.get(frozenset([obs1, obs2])):
            return True
        for par in covparameters:
            if par.influences(obs1) and par.influences(obs2):
                return True
        return False

    def sortobservables(self, observables, covparameters):
        """Splits observables into such a groups that observables that are
        all correlated with respect to a covparameters or pull terms

        If observables are not provided: use ones stored in data
        """
        if observables is None:
            observables = self.data.keys()

        to_process = list(observables)
        groups = [[]]
        while to_process:
            for obs in to_process:
                if not groups[-1] or any(self.iscorrelated(obs, x, covparameters) for x in groups[-1]):
                    groups[-1].append(obs)
                    break
            else:
                groups.append([])
                continue
            to_process.remove(obs)
        groups = [sorted(group, key=lambda t: observables.index(t)) for group in groups]
        return sorted(groups, key=lambda t: observables.index(t[0]))

    def _group_covpars(self, covparameters):
        to_process = list(covparameters)
        groups = [[]]
        while to_process:
            for par in to_process:
                if not par.isCorrelated:
                    continue
                if not groups[-1] or any(par.isCorrelated(x) for x in groups[-1]):
                    groups[-1].append(par)
                    break
            else:
                groups.append([])
                continue
            to_process.remove(obs)
        groups = [sorted(group, key=lambda t: covparameters.index(t)) for group in groups]
        return sorted(groups, key=lambda t: covparameters.index(t[0]))

    def assigncovariances(self, observables, prediction, covparameters):
        """Assign covariance for a given observable and add it to a
        prediction. Checks for covariance that is already in the
        self.covariance and than adds it computes derivatives of observable
        with respect to all covparameters and put it into prediction
        """
        for covkey, covs in self.covariance.iteritems():
            if len(covkey & observables) != len(covkey):
                continue
            if len(covkey) == 1:
                obs1 = obs2 = list(covkey)[0]
            else:
                obs1, obs2 = list(covkey)
            for cov in covs:
                prediction.covariate(cov, obs1, 1, obs2, 1)
        if any(par.influences(prediction.prediction) for par in covparameters):
            jac = ROOT.Jacobian()
            par_covs = ROOT.ParCovMatrix()
            for par in covparameters:
                jac.append(par)
                par_covs.append(par)
            prediction.prediction_ready()
            par_covs.materialize()
            jac.jacobian.func(prediction.prediction)
            jac_T = ROOT.Transpose()
            jac_T.transpose.mat(jac.jacobian)
            product = ROOT.MatrixProduct()
            product.multiply(jac.jacobian)
            product.multiply(par_covs.unc_matrix)
            product.multiply(jac_T.transpose.T)
            product.product.touch() # Fixme: should be controllable
            prediction.addSystematicCovMatrix(product.product)
        prediction.finalize()

    def makedata(self, obsblock):
        """ Returns either observable itself or a concantenation of
        values of observables
        """
        datas = [self.data.get(obs) for obs in obsblock]
        if any(data is None for data in datas):
            return None
        if len(datas) == 1:
            return datas[0]
        merged = ROOT.Concat()
        for data in datas:
            merged.append(data)
        return merged

    def makeblocks(self, observables, covparameters):
        """ Returns a list of blocks with theoretical prediction, data and
        covariance matrix.

        Accepts list of observables that would be used as theoretical
        prediction and parameters for
        what covariance matrix is going to be calculated.
        """
        blocks = []
        for i, obsblock in enumerate(self.sortobservables(observables, covparameters)):
            prediction = ROOT.CovariatedPrediction()
            for obs in obsblock:
                prediction.append(obs)
            self.assigncovariances(set(obsblock), prediction, covparameters)
            data = self.makedata(obsblock)
            if data is None:
                raise Exception("no data constructed")
            prediction.cov.Lbase(prediction.covbase.L)

            if len(obsblock)>1:
                blocks.append(Block(prediction.prediction, data, prediction.cov))
            else:
                blocks.append(Block(obsblock[0].single(), data, prediction.cov))
        return blocks
