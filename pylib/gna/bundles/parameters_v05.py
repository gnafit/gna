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
    skip = ['meta', 'uncertainty', 'uncertainty_mode']
    def __init__(self, *args, **kwargs):
        self._par_container = []
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        if 'state' in self.cfg and not self.cfg.state in ('fixed', 'free'):
            raise ValueError('Invalid state: '+self.cfg.state)

    @classmethod
    def _provides(cls, cfg):
        pars = cfg_parse(cfg.pars, verbose=False)
        names = list(pars.keys())
        skips = list(cfg.get('skip', ()))+cls.skip
        for skip in skips:
            try:
                names.remove(skip)
            except ValueError:
                pass
        return names, names if cfg.get('objectize') else ()

    def get_parameter_kwargs(self, parcfg, state, uncertainty_common, uncertainty_mode_common):
        kwargs=dict()

        if isinstance(parcfg, Iterable):
            parcfg=list(parcfg)
        else:
            parcfg=[parcfg]

        uncertainty_mode = None
        if len(parcfg)==1:
            kwargs['central'] = parcfg[0]

            if uncertainty_common is not None:
                err = uncertainty_common
                uncertainty_mode = uncertainty_mode_common
            else:
                kwargs[state] = True
        else:
            if len(parcfg)==2:
                kwargs['central'], err = parcfg
                uncertainty_mode = uncertainty_mode_common
            else:
                kwargs['central'], err, uncertainty_mode = parcfg[:3]

        if uncertainty_mode is not None:
            if uncertainty_mode=='absolute':
                kwargs['sigma'] = err
            elif uncertainty_mode=='relative':
                kwargs['relsigma'] = err
            elif uncertainty_mode=='percent':
                kwargs['relsigma'] = err*0.01

        return kwargs

    def define_variables(self):
        self._par_container = []
        pars = cfg_parse(self.cfg.pars, verbose=True)
        labels = self.cfg.get('labels', pars.get('meta', {}).get('labels', {}))
        objectize = self.cfg.get('objectize')
        skip = list(self.cfg.get('skip', ()))+self.skip


        state = self.cfg.get('state', None)
        uncertainty_common = pars.get('uncertainty', None)
        uncertainty_mode_common = pars.get('uncertainty_mode', 'absolute')
        for name, parcfg in pars.items():
            if name in skip:
                continue

            kwargs=self.get_parameter_kwargs(parcfg, state=state, uncertainty_common=uncertainty_common, uncertainty_mode_common=uncertainty_mode_common)
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
