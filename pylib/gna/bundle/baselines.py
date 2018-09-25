# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as np
import constructors as C
from gna.bundle import *
import itertools

class baselines(TransformationBundle):
    def __init__(self, **kwargs):
        TransformationBundle.__init__( self, **kwargs )

        #Handle both indexed and indexless cases
        try:
            self.idx = self.cfg.indices
            from gna.expression import NIndex
            if not isinstance(self.idx, NIndex):
                self.idx = NIndex(fromlist=self.cfg.indices)
            # Check that naming in index and data is consistent
            self.constistency_check() 
        except KeyError:
            print("No indices is passed")

        print(kwargs)
        self.detectors = kwargs['cfg']['detectors']
        print(self.detectors)
        self.reactors = kwargs['cfg']['reactors']


    def build(self):
        '''So empty lol'''
        pass

    def constistency_check(self):
        '''Checks that sets of detectors and reactor coincide exactly in
        from indices and from kwargs'''

        raise NotImplementedError

        #  if self.idx is None:
            #  return True

        #  from_kwargs = set(self.reactors.keys(), self.detectors.keys())
        #  reactor_idx, det_idx = self.idx.names().index('reactor'), self.idx.names().index('detector')

        


    def compute_distance(self, reactor, detector):
        '''Computes distance between pair of reactor and detector. Coordinates
        of both of them should have the same shape, i.e. either 1d or 3d'''

        assert len(reactor) == len(detector), "Dimensions of reactor and detector doesn't match"

        return np.sqrt(np.sum((np.array(reactor) - np.array(detector))**2))

    def define_variables(self):
        '''Create baseline variables in a common_namespace'''

        for reactor, detector in itertools.product(self.reactors.items(), self.detectors.items()):
            print(reactor, detector)
            distance = self.compute_distance(reactor[1], detector[1])
            print(distance)
            key = "baseline.{}.{}".format(detector[0], reactor[0])
            print(key)
            self.common_namespace.reqparameter(key, central=distance, sigma=0.1, fixed=True)
