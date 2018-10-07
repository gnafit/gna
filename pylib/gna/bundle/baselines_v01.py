# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as np
import constructors as C
from gna.bundle import *
from gna.configurator import NestedDict
import itertools

class baselines(TransformationBundle):
    def __init__(self, **kwargs):
        TransformationBundle.__init__( self, **kwargs )

        self.init_indices()
        self.init_data()

        #Handle both indexed and indexless cases

    def build(self):
        '''Deliberately empty'''
        pass

    def init_data(self):
        '''Read configurations of reactors and detectors from either files or
        dicts'''

        from gna.configurator import configurator
        def get_data(source):
            if isinstance(source, str):
                try:
                    data = configurator(source)
                    return data['data']
                except:
                    raise Exception('Unable to open or parse file {}'.format(source) ) 
            elif isinstance(source, NestedDict):
                return source

        self.detectors = get_data(self.cfg.detectors)
        self.reactors  = get_data(self.cfg.reactors)
        if any('AD' not in str(key) for key in self.detectors.keys()):
            print('AD is not in detectors keys! Substituting')
            new = {}
            for key, value in self.detectors.items():
                if 'AD' not in str(key):
                    new['AD'+str(key)] = value
                else:
                    new[str(key)] = value
            self.detectors = new
                
                


    def compute_distance(self, reactor, detector):
        '''Computes distance between pair of reactor and detector. Coordinates
        of both of them should have the same shape, i.e. either 1d or 3d'''

        assert len(reactor) == len(detector), "Dimensions of reactor and detector doesn't match"

        return np.sqrt(np.sum((np.array(reactor) - np.array(detector))**2))

    def define_variables(self):
        '''Create baseline variables in a common_namespace'''

        for i, it in enumerate(self.idx.iterate()):
            name = it.current_format('{name}{autoindex}', name='baseline')
            cur_det, cur_reactor = it.get_current('d'), it.get_current('r')
            try:
                detector, reactor = self.detectors[cur_det], self.reactors[cur_reactor]
            except KeyError as e:
                msg = "Detector {det} or reactor {reac} are missing in the configuration"
                raise KeyError, msg.format(det=cur_det, reac=cur_reactor)


            distance = self.compute_distance(reactor=reactor, detector=detector)
            self.common_namespace.reqparameter(name, central=distance,
                    sigma=0.1, fixed=True, label="Baseline between {} and {}, m".format(cur_det, cur_reactor))

            inv_key = it.current_format("{name}{autoindex}", name='baselineweight')
            self.common_namespace.reqparameter(inv_key, central=0.25/distance**2/np.pi, sigma=0.1, fixed=True,
                        label="1/(4πL²) for {} and {}, m⁻²".format(cur_det, cur_reactor))
