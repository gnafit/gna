from __future__ import print_function
import ROOT
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
            #  self.covariance.update(base.covariance)
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
                                 and not key.isFixed()]
        print("Underlying params -- {}".format(underlaying_params))
        for (par_1, base_1), (par_2, base_2) in combinations(underlaying_params, 2):
            mutual_cov =  par_1.getCovariance(par_2)
            if par_1 != par_2:
                print(mutual_cov, "mutual cov")
            self.covariance[frozenset([par_1, par_2])] = [self._pointize(mutual_cov)]
        print(self.covariance)

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
            obj = ROOT.Points(obj)
        return obj

    def covariate(self, obs1, obs2, cov):
        """ Given two observables and covariance between them, add covariate
        into a stotage for covariances
        """
        self.covariance[frozenset([obs1, obs2])].append(self._pointize(cov))

    def iscovariated(self, obs1, obs2, covparameters):
        """Checks whether two observables are covariated or affected by covparameters
        """
        print(frozenset([obs1, obs2]), obs1, obs2)
        if self.covariance.get(frozenset([obs1, obs2])):
            print("Yes, I'm already in the covlist!!!!!!!")
            return True
        for par in covparameters:
            if par.influences(obs1) and par.influences(obs2):
                return True
        return False

    def sortobservables(self, observables, covparameters):
        """Splits observables into such a groups that observables that are
        all covariated with respect to a covparameters or pull terms 
        """
        to_process = list(observables)
        groups = [[]]
        while to_process:
            for obs in to_process:
                if not groups[-1] or any(self.iscovariated(obs, x, covparameters) for x in groups[-1]):
                    groups[-1].append(obs)
                    break
            else:
                groups.append([])
                continue
            to_process.remove(obs)
        groups = [sorted(group, key=lambda t: observables.index(t)) for group in groups]
        return sorted(groups, key=lambda t: observables.index(t[0]))

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
            print('obs1 -- {0}, obs2 -- {1}'.format(obs1, obs2))
            for cov in covs:
                print('cov', cov)
                prediction.covariate(cov, obs1, 1, obs2, 1)
        for par in covparameters:
            der = ROOT.Derivative(par)
            der.derivative.inputs(prediction.prediction)
            prediction.rank1(der)
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
        merged = ROOT.Prediction()
        for data in datas:
            merged.append(data)
        return merged

    def makeblocks(self, observables, covparameters):
        """ Returns a list of blocks with theoretical prediction, data and
        covariance matrix.

        Accepts list of observables that would be used as theory and parameters for
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
            blocks.append(Block(prediction.prediction, data, prediction.cov))
        return blocks
