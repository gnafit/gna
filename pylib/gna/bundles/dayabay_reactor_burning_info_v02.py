# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from load import ROOT as R
import numpy as np
import gna.constructors as C
from gna.bundle import TransformationBundle
from collections import defaultdict
from gna.configurator import configurator

mapping_idx_to_reactors = {1: "DB1", 2: "DB2", 3: "LA1", 4: "LA2", 5: "LA3", 6: "LA4"}

class dayabay_reactor_burning_info_v02(TransformationBundle):
    def __init__(self, *args, **kwargs):
        '''Initialize reactor information such as daily ratios of actual
        thermal power to nominal, fission fractions per core.
        Cores are provided by caller through indices.
        Info is read from npz file.
        '''
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(2, 2, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        self.nidx_reactor = self.nidx.get_subset(self.bundlecfg.major[0])
        self.nidx_isotope = self.nidx.get_subset(self.bundlecfg.major[1])

        self.init_data()

    @staticmethod
    def _provides(cfg):
        if not cfg.add_ff:
            return ('fission_fraction_corr',), ('thermal_power', 'fission_fractions',)
        else:
            return ('fission_fraction_corr',), ('thermal_power', 'fission_fractions', 'fission_fractions_add')

    def init_data(self):
        try:
            self.file_idx_to_reactor = self.cfg.name_mapping
        except KeyError:
            self.file_idx_to_reactor = mapping_idx_to_reactors

        self.core_info_daily = defaultdict(lambda: defaultdict(list))
        with np.load(self.cfg.reactor_info) as f:
            self.data = f['power']

            seconds_per_day = 60*60*24
            for it in self.data:
                current_core = self.core_info_daily[ self.file_idx_to_reactor[it['core']] ]
                current_core['power'].append(it['power'])
                current_core['days'].append((it['end'] - it['start'])/seconds_per_day)
                current_core['fission_fractions'].append(it['iso'])

        self.fission_info = configurator(self.cfg.fission_uncertainty_info)

    def define_variables(self):
        isotopes=self.fission_info['isotopes']
        relsigmas=self.fission_info['relsigma']
        for reacit in self.nidx_reactor:
            isotope_pac = []
            for isoit in self.nidx_isotope:
                it = reacit + isoit
                iso_name,  = isoit.current_values()
                reac_name, = reacit.current_values()

                iso_idx = isotopes.index(iso_name)
                relsigma = relsigmas[iso_idx]
                ff_name = it.current_format(name='fission_fraction_corr')
                label="Fission fraction of isotope {iso} in reactor {reac}".format(iso=iso_name, reac=reac_name)

                isotope_pac.append((ff_name, {'central': 1, 'relsigma': relsigma, 'label': label,}))

            # check the order of isotopes matches corresponding order from
            # configuration
            if all(from_conf in from_pack[0] for from_pack, from_conf
                   in zip(isotope_pac, self.fission_info['isotopes'])):
                if len(isotope_pac) == len(self.fission_info['isotopes']):
                    self.namespace.reqparameter_group(*isotope_pac, **{'covmat': self.fission_info['correlation']})
                else:
                    self.namespace.reqparameter_group(*isotope_pac)

            else:
                raise Exception("Orderings of isotopes from indices and data "
                                "do not match. Check it!")

    def build(self):
        for rit in self.nidx_reactor:
            core_name, = rit.current_values()
            core = self.core_info_daily[core_name]
            days_in_period = np.array(core['days'])

            thermal_power_daily = np.repeat(core['power'], days_in_period)
            if self.cfg.nominal_power:
                thermal_power_daily = np.ones(len(thermal_power_daily))
            thermal_power_per_core = C.Points(thermal_power_daily, labels=rit.current_format('Thermal power\n{autoindex}'))
            self.context.objects[(core_name, 'thermal_power')]  = thermal_power_per_core
            self.set_output('thermal_power', rit, thermal_power_per_core.single())

            for oit in self.nidx_isotope:
                it = rit+oit
                iso_name, = oit.current_values()

                # map fission fractions and thermal powers to days instead of weeks
                fission_fractions_daily = np.repeat(core['fission_fractions'], days_in_period)
                label=it.current_format('Fission fractions\n{autoindex}')
                fission_per_iso = C.Points(fission_fractions_daily[iso_name], labels=label)
                self.context.objects[it.current_values(name='fission_fractions')] = fission_per_iso
                self.set_output('fission_fractions', it, fission_per_iso.single())
                if self.cfg.add_ff:
                    label_add=it.current_format('Additional set of fission fractions for norm calc\n{autoindex}')
                    fission_per_iso_add = C.Points(fission_fractions_daily[iso_name], labels=label_add)
                    self.context.objects[it.current_values(name='fission_fractions_add')] = fission_per_iso_add
                    self.set_output('fission_fractions_add', it, fission_per_iso_add.single())
