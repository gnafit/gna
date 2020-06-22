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

        if 'mode' in self.cfg and not self.cfg.mode in ('fixed', 'free'):
            raise ValueError('Invalid mode: '+self.cfg.mode)

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

    def define_variables(self):
        self._par_container = []
        pars = cfg_parse(self.cfg.pars, verbose=True)
        labels = self.cfg.get('labels', pars.get('labels', {}))
        objectize = self.cfg.get('objectize')
        skip = self.cfg.get('skip', ())

        mode = self.cfg.get('mode', None)
        for name, parcfg in pars.items():
            if name in skip or name=='labels':
                continue
            kwargs=dict()
            if isinstance(parcfg, Iterable):
                parcfg=list(parcfg)
                if len(parcfg)==1:
                    kwargs['central'] = parcfg[0]
                    kwargs[mode] = True
                elif len(parcfg)==2:
                    kwargs['central'], kwargs['sigma'] = parcfg
                elif len(parcfg)==3:
                    kwargs['central'], err, mode = parcfg
                    if mode=='absolute':
                        kwargs['sigma'] = err
                    elif mode=='relative':
                        kwargs['relsigma'] = err
                    elif mode=='percent':
                        kwargs['relsigma'] = err*0.01
            else:
                kwargs['central'] = parcfg
                kwargs[mode] = True

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
