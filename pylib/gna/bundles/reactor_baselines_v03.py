# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as np
import gna.constructors as C
from gna.bundle import *
from collections import OrderedDict, Iterable
import itertools
from tools.cfg_load import cfg_parse

Units = R.NeutrinoUnits

conversion = {"meter": 1.e-3, "kilometer": 1.0}
conversion['m']=conversion['meter']
conversion['km']=conversion['kilometer']

class reactor_baselines_v03(TransformationBundle):
    """Calcualte reactor baselines v03:
       Changes since v02:
       - Enable <2 major indices
       Changes since v01:
       - enable yaml_input
       - do not add 'AD' to the beginning of the detector name
        """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 2, 'major')
        self.init_data()

        if not self.cfg.unit in conversion:
            print('Available units:', *conversion.keys())
            raise Exception('Invalid unit: '+self.cfg.unit)

    @staticmethod
    def _provides(cfg):
        return ('baseline', 'baselineweight'), ()

    def init_data(self):
        '''Read configurations of reactors and detectors from either files or dicts'''
        self.detectors = cfg_parse(self.cfg.detectors)
        self.reactors  = cfg_parse(self.cfg.reactors)

        if 'reactors_key' in self.cfg:
            self.reactors = self.reactors[self.cfg['reactors_key']]

        if 'detectors_key' in self.cfg:
            self.detectors = self.detectors[self.cfg['detectors_key']]

    def compute_distance(self, reactor, detector):
        '''Computes distance between pair of reactor and detector. Coordinates
        of both of them should have the same shape, i.e. either 1d or 3d'''

        if not isinstance(reactor, Iterable):
            reactor = [reactor]
        if not isinstance(detector, Iterable):
            detector = [detector]

        assert len(reactor) == len(detector), "Dimensions of reactor and detector doesn't match"

        return conversion[self.cfg.unit]*(np.sqrt(np.sum((np.array(reactor) - np.array(detector))**2)))

    def define_variables(self):
        reactor_name, detector_name = self.cfg.bundle.major
        label_b = self.cfg.get('label', "Baseline between {detector} and {reactor}, km")
        label_bw =self.cfg.get('label_w', "1/(4πL²) for {detector} and {reactor}, cm⁻²")
        for i, it_major in enumerate(self.nidx_major):
            cur_reactor, cur_det = it_major.get_current(reactor_name), it_major.get_current(detector_name)
            try:
                reactor, detector = self.reactors[cur_reactor], self.detectors[cur_det]
            except KeyError as e:
                msg = "Detector {det} or reactor {reac} are missing in the configuration"
                raise KeyError, msg.format(det=cur_det, reac=cur_reactor)

            distance = self.compute_distance(reactor=reactor, detector=detector)
            const = 0.25/np.pi*1.e-10 # Convert 1/km2 to 1/cm2
            for it_minor in self.nidx_minor:
                it = it_minor+it_major
                self.reqparameter('baseline', it, central=distance,
                                  sigma=0.1, fixed=True, label=label_b.format(detector=cur_det, reactor=cur_reactor))

                self.reqparameter('baselineweight', it, central=const/distance**2, fixed=True,
                                  label=label_bw.format(detector=cur_det, reactor=cur_reactor))

