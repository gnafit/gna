# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.bundle.bundle import *

class parameters_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

    @staticmethod
    def _provides(cfg):
        return (cfg.parameter), ()

    def define_variables(self):
        separate_uncertainty = self.cfg.get('separate_uncertainty', False)
        parname = self.cfg.parameter
        pars = self.cfg.pars
        labelfmt = self.cfg.get('label', '')

        for it_major in self.nidx_major:
            major_values = it_major.current_values()
            if major_values:
                parcfg = pars[major_values]
            else:
                parcfg = pars

            for it_minor in self.nidx_minor:
                it=it_major+it_minor
                label = it.current_format(labelfmt) if labelfmt else ''

                if separate_uncertainty:
                    if parcfg.mode=='fixed':
                        raise self.exception('Can not separate uncertainty for fixed parameters')

                    unccfg = parcfg.get_unc()
                    uncpar = self.reqparameter(separate_uncertainty, it, cfg=unccfg, label=label+' (norm)')
                    parcfg.mode='fixed'

                par = self.reqparameter(parname, it, cfg=parcfg, label=label)
