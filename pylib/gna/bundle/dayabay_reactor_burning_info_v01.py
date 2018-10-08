# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as np
import constructors as C
from gna.bundle import *
from collections import defaultdict

mapping_idx_to_reactors = {1: "DB1", 2: "DB2", 3: "LA1", 4: "LA2", 5: "LA3", 6: "LA4"}

class dayabay_reactor_burning_info_v01(TransformationBundle):
    def __init__(self, **kwargs):
        super(dayabay_reactor_burning_info_v01, self).__init__( **kwargs )

        self.init_indices()
        if not self.idx.ndim()==1:
            raise self.exception('require 1 index exactly')
        self.init_data()

    def init_data(self):
        try:
            self.file_idx_to_reactor = self.cfg.name_mapping
        except KeyError:
            self.file_idx_to_reactor = mapping_idx_to_reactors

        self.core_info_daily = defaultdict(lambda: defaultdict(list))
        with np.load(self.cfg.reactor_info) as f:
            self.data = f['power']
            sec_per_day = 60*60*24
            for it in self.data:
                current_core = self.core_info_daily[ self.file_idx_to_reactor[it['core']] ]
                current_core['power'].append(it['power'])
                current_core['days'].append((it['end'] - it['start'])/sec_per_day)
                current_core['fission_fractions'].append(it['iso'])
            

    def define_variables(self):
        pass

    def build(self):
        #  import IPython
        #  IPython.embed()
        for idx in self.idx:    
            isotopes = defaultdict(np.array)
            core_name, = idx.current_values()
            core = self.core_info_daily[core_name]
            days_in_period = np.array(core['days'])

            # replicate thermal powers times corresponding to reactor working time
            thermal_power_daily = np.repeat(core['power'], days_in_period)
            fission_fractions_daily = np.repeat(core['fission_fractions'], days_in_period)
            thermal_power_per_core = C.Points(thermal_power_daily)
            self.objects[(core_name, 'ThermalPower')]  = thermal_power_per_core
            self.set_output(thermal_power_per_core.single(), "thermalpower", idx)
            for iso in fission_fractions_daily.dtype.names:
                fission_per_iso = C.Points(fission_fractions_daily[iso])
                self.objects[(core_name, 'fission_fractions', iso)] = fission_per_iso
                self.set_output(fission_per_iso.single(), "fission_fractions.{}".format(iso), idx)
             




        




