# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from load import ROOT as R
import numpy as np
import gna.constructors as C
from gna.bundle import *
from gna.configurator import NestedDict
from collections import OrderedDict
import itertools

Units = R.NeutrinoUnits

conversion = {"meter": 1.e-3, "kilometer": 1.0}
conversion['m']=conversion['meter']
conversion['km']=conversion['kilometer']

class reactor_baselines_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(2, 2, 'major')
        self.init_data()

        if not self.cfg.unit in conversion:
            print('Available units:', *conversion.keys())
            raise Exception('Invalid unit: '+self.cfg.unit)

    @staticmethod
    def _provides(cfg):
        return ('baseline', 'baselineweight'), ()

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
                   print('Unable to open or parse file {}'.format(source) )
                   raise
            elif isinstance(source, (NestedDict, OrderedDict, dict)):
                return source
            else:
                raise Exception("Wrong type of data source {}".format(type(source)))

        self.detectors = get_data(self.cfg.detectors)
        self.reactors  = get_data(self.cfg.reactors)

        snf_pools = self.cfg.get('snf_pools', None)
        self.snf_pools = snf_pools and get_data(snf_pools) or None

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

        return conversion[self.cfg.unit]*(np.sqrt(np.sum((np.array(reactor) - np.array(detector))**2)))

    def define_variables(self):
        reactor_name, detector_name = self.cfg.bundle.major
        for i, it_major in enumerate(self.nidx_major):
            cur_reactor, cur_det = it_major.get_current(reactor_name), it_major.get_current(detector_name)
            try:
                reactor, detector = self.reactors[cur_reactor], self.detectors[cur_det]
            except KeyError as e:
                msg = "Detector {det} or reactor {reac} are missing in the configuration"
                raise KeyError(msg.format(det=cur_det, reac=cur_reactor))

            distance = self.compute_distance(reactor=reactor, detector=detector)
            const = 0.25/np.pi*1.e-10 # Convert 1/km2 to 1/cm2
            for it_minor in self.nidx_minor:
                it = it_minor+it_major
                self.reqparameter('baseline', it, central=distance,
                        sigma=0.1, fixed=True, label="Baseline between {} and {}, km".format(cur_det, cur_reactor))

                self.reqparameter('baselineweight', it, central=const/distance**2, fixed=True,
                                  label="1/(4πL²) for {} and {}, cm⁻²".format(cur_det, cur_reactor))
