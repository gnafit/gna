# -*- coding: utf-8 -*-

from __future__ import print_function

from gna.bundle.bundle import TransformationBundleV01

class parameters_ex01(TransformationBundleV01):
    def __init__(self, *args, **kwargs):
        TransformationBundleV01.__init__(self, *args, **kwargs)

    def define_variables(self):
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

                par = self.reqparameter(parname, it, cfg=parcfg, label=label)
