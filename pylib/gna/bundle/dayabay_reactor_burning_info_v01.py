# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as np
import constructors as C
from gna.bundle import *
from collections import defaultdict
from gna.configurator import configurator

mapping_idx_to_reactors = {1: "DB1", 2: "DB2", 3: "LA1", 4: "LA2", 5: "LA3", 6: "LA4"}

class dayabay_reactor_burning_info_v01(TransformationBundle):
    def __init__(self, **kwargs):
        '''Initialize reactor information such as daily ratios of actual
        thermal power to nominal, fission fractions per core.
        Cores are provided by caller through indices.
        Info is read from npz file.
        '''
        super(dayabay_reactor_burning_info_v01, self).__init__( **kwargs )

        self.init_indices()
        if not self.idx.ndim()==2:
            raise self.exception('Need exactly 2 indices. {} passed'.format(self.idx.ndim()))

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
        reac, other = self.idx.split( ('r') )
        for rit in reac.iterate():
            core_name, = rit.current_values()
            core = self.core_info_daily[core_name]
            days_in_period = np.array(core['days'])

            thermal_power_daily = np.repeat(core['power'], days_in_period)
            thermal_power_per_core = C.Points(thermal_power_daily)
            thermal_power_per_core.points.setLabel( rit.current_format('Thermal power\n{autoindexnd}') )
            self.objects[(core_name, 'thermal_power')]  = thermal_power_per_core
            self.set_output(thermal_power_per_core.single(), 'thermal_power', rit)

            for oit in other.iterate():
                it = rit+oit
                iso_name = it.indices['i'].current

                # map fission fractions and thermal powers to days instead of weeks
                fission_fractions_daily = np.repeat(core['fission_fractions'], days_in_period)
                fission_per_iso = C.Points(fission_fractions_daily[iso_name])
                fission_per_iso.points.setLabel( it.current_format('Fission fractions\n{autoindexnd') )
                self.objects[(core_name, 'fission_fractions', iso_name)] = fission_per_iso
                self.set_output(fission_per_iso.single(), 'fission_fractions', it)
