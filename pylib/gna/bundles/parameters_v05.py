# -*- coding: utf-8 -*-

"""Parameters v05 bundle
Implements a set of unrelated parameters.

Unlike v04, does not depend on uncertaindict

Based on: parameters_v04
"""

from __future__ import print_function
from load import ROOT as R
from gna.bundle.bundle import *
import numpy as N
from gna import constructors as C
from tools.cfg_load import cfg_parse
from collections import Iterable

class parameters_v05(TransformationBundle):
    covmat, corrmat = None, None
    def __init__(self, *args, **kwargs):
        self._par_container = []
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        if 'state' in self.cfg and not self.cfg.state in ('fixed', 'free'):
            raise ValueError('Invalid state: '+self.cfg.state)

    @staticmethod
    def _provides(cfg):
        pars = cfg_parse(cfg.pars, verbose=False)
        names = list(pars.keys())
        skips = list(cfg.get('skip', ()))+['labels']
        for skip in skips:
            try:
                names.remove(skip)
            except ValueError:
                pass
        return names, names if cfg.get('objectize') else ()

    def get_parameter_kwargs(self, parcfg, state, uncertainty_mode_common):
        kwargs=dict()
        if isinstance(parcfg, Iterable):
            parcfg=list(parcfg)
            if len(parcfg)==1:
                kwargs['central'] = parcfg[0]
                kwargs[state] = True
            else:
                if len(parcfg)==2:
                    kwargs['central'], kwargs['sigma'] = parcfg
                    uncertainty_mode = uncertainty_mode_common
                elif len(parcfg)==3:
                    kwargs['central'], err, uncertainty_mode = parcfg

                if uncertainty_mode=='absolute':
                    kwargs['sigma'] = err
                elif uncertainty_mode=='relative':
                    kwargs['relsigma'] = err
                elif uncertainty_mode=='percent':
                    kwargs['relsigma'] = err*0.01
        else:
            kwargs['central'] = parcfg
            kwargs[state] = True

        return kwargs

    def define_variables(self):
        self._par_container = []
        pars = cfg_parse(self.cfg.pars, verbose=True)
        labels = self.cfg.get('labels', pars.get('labels', {}))
        objectize = self.cfg.get('objectize')
        skip = self.cfg.get('skip', ())


        state = self.cfg.get('state', None)
        uncertainty_mode_common = self.cfg.get('uncertainty_mode', 'absolute')
        for name, parcfg in pars.items():
            if name in skip or name=='labels':
                continue

            kwargs=self.get_parameter_kwargs(parcfg, state=state, uncertainty_mode_common=uncertainty_mode_common)
            if name in labels:
                kwargs['label'] = labels[name]
            par = self.reqparameter(name, None, **kwargs)

            if objectize:
                trans=par.transformations.value
                trans.setLabel(label)
                self.set_output(name, it, trans.single())

                self._par_container.append(par)

    def build(self):
        pass
