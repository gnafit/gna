# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as np
import gna.constructors as C
from gna.bundle import *
import itertools

class baselines(TransformationBundleLegacy):
    def __init__(self, *args, **kwargs):
        TransformationBundleLegacy.__init__( self, *args, **kwargs )

        self.detectors = kwargs['cfg']['detectors']
        self.reactors = kwargs['cfg']['reactors']
        #Handle both indexed and indexless cases
        try:
            self.idx = self.cfg.indices
            from gna.expression import NIndex
            if not isinstance(self.idx, NIndex):
                self.idx = NIndex(fromlist=self.cfg.indices)

            # Check that naming in index and data is consistent
            self.constistency_check()
        except KeyError:
            pass




    def build(self):
        '''Deliberately empty'''
        pass

    def constistency_check(self):
        '''Checks that sets of detectors and reactor coincide exactly in
        from indices and from kwargs'''


        if self.idx is None:
            return True
        from itertools import chain

        from_kwargs = set(chain(self.reactors.keys(), self.detectors.keys()))
        from_idx = set(chain.from_iterable((i.variants for i in self.idx.indices.itervalues())))
        if from_idx == from_kwargs:
            return True
        else:
            raise Exception("Reactors and detectors in indices and from "
                    "configuration does not match.\n From indices {} \n and "
                    "from configuration {}".format(from_idx, from_kwargs))

    def compute_distance(self, reactor, detector):
        '''Computes distance between pair of reactor and detector. Coordinates
        of both of them should have the same shape, i.e. either 1d or 3d'''

        assert len(reactor) == len(detector), "Dimensions of reactor and detector doesn't match"

        return np.sqrt(np.sum((np.array(reactor) - np.array(detector))**2))

    def define_variables(self):
        '''Create baseline variables in a common_namespace'''

        for reactor, detector in itertools.product(self.reactors.items(), self.detectors.items()):
            distance = self.compute_distance(reactor[1], detector[1])
            key = "baseline.{}.{}".format(detector[0], reactor[0])
            self.common_namespace.reqparameter(key, central=distance,
                    sigma=0.1, fixed=True, label="Baseline between {} and {}".format(reactor[0], detector[0]))
