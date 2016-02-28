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
            self.covariance.update(base.covariance)

    def assign(self, obs, value, error):
        self.data[obs] = self._pointize(value)
        self.covariate(obs, obs, error)

    def _pointize(self, obj):
        if not isinstance(type(obj), ROOT.PyRootType):
            obj = ROOT.Points(obj)
        return obj

    def covariate(self, obs1, obs2, cov):
        self.covariance[frozenset([obs1, obs2])].append(self._pointize(cov))

    def iscovariated(self, obs1, obs2, covparameters):
        if self.covariance.get(frozenset([obs1, obs2])):
            return True
        for par in covparameters:
            if par.affects(obs1) and par.affects(obs2):
                return True
        return False

    def sortobservables(self, observables, covparameters):
        toprocess = list(observables)
        groups = [[]]
        while toprocess:
            for obs in toprocess:
                if not groups[-1] or any(self.iscovariated(obs, x, covparameters) for x in groups[-1]):
                    groups[-1].append(obs)
                    break
            else:
                groups.append([])
                continue
            toprocess.remove(obs)
        groups = [sorted(group, key=lambda t: observables.index(t)) for group in groups]
        return sorted(groups, key=lambda t: observables.index(t[0]))

    def assigncovariances(self, observables, prediction, covparameters):
        for covkey, covs in self.covariance.iteritems():
            if len(covkey & observables) != len(covkey):
                continue
            if len(covkey) == 1:
                obs1 = obs2 = list(covkey)[0]
            else:
                obs1, obs2 = list(covkey)
            for cov in covs:
                prediction.covariate(cov, obs1, 1, obs2, 1)
        for par in covparameters:
            der = ROOT.Derivative(par)
            der.derivative.inputs(prediction.prediction)
            prediction.rank1(der)
        prediction.finalize()

    def makedata(self, obsblock):
        datas = [self.data.get(obs) for obs in obsblock]
        if any(data is None for data in datas):
            return None
        if len(datas) == 1:
            return datas[0]
        merged = np.hstack([np.ravel(data) for data in datas])
        return ROOT.Points(merged)
            
    def makeblocks(self, observables, covparameters):
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
