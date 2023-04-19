"""Parameters v06 bundle
Implements a set of parameters, defined via
- dictionary
- python file (as dictionary)
- yaml file (as dictionary)

Based on: parameters_v05

Implements:
- Indices
   + major: the distinct value is read from the configuration
   + minor: replicas
- Separate uncertainty
"""

from load import ROOT as R
from gna.bundle.bundle import *
import numpy as N
from gna import constructors as C
from tools.cfg_load import cfg_parse
from collections.abc import Iterable, Mapping
from tools.dictwrapper import DictWrapper

class parameters_v06(TransformationBundle):
    covmat, corrmat = None, None
    skip = ['meta', 'uncertainty', 'uncertainty_mode', 'state']
    def __init__(self, *args, **kwargs):
        self._par_container = []
        TransformationBundle.__init__(self, *args, **kwargs)
        # self.check_nidx_dim(0, 0, 'major')
        # self.check_nidx_dim(0, 0, 'minor')

        if 'state' in self.cfg and not self.cfg.state in ('fixed', 'free'):
            raise ValueError('Invalid state: '+self.cfg.state)

        nsname = self.cfg.get('namespace')
        if nsname:
            self.namespace = self.namespace(nsname)

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

        extra = cfg.get('hooks', {}).keys()
        names = names + list(extra)

        sepuncfmt=cfg.get('separate_uncertainty')
        if sepuncfmt:
            names.extend(tuple(map(sepuncfmt.format, names)))

        if cfg.get('objectize'):
            onames = tuple(names)
        else:
            onames = ()

        nsname = cfg.get('namespace')
        if nsname:
            names = tuple('.'.join((nsname, n)) for n in names)

        return names, onames

    def get_parameter_kwargs(self, pars, name, idx, state, uncertainty_common, uncertainty_mode_common):
        kwargs=dict()
        try:
            parcfg=pars[name, idx]
        except KeyError:
            raise Exception('Unable to get parameter configuration for {}.{}'.format(name, idx))

        if isinstance(parcfg, DictWrapper):
            raise Exception('Invalid index specified for {}.{}'.format(name, idx))

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

    def get_scale_kwargs(self, name, kwargspar):
        sepuncfmt = self.cfg.get('separate_uncertainty')

        if not sepuncfmt:
            return None, None

        if kwargspar.get('fixed') or kwargspar.get('free'):
            return None, None

        kwargsscale = kwargspar.copy()
        if 'label' in kwargsscale:
            kwargsscale['label'] = 'Scale for '+kwargsscale['label']

        if 'relsigma' in kwargsscale:
            kwargsscale['sigma'] = kwargsscale['relsigma']
            del kwargsscale['relsigma']
            del kwargspar['relsigma']
        elif 'percent' in kwargsscale:
            kwargsscale['sigma'] = kwargsscale['percent']*0.01
            del kwargsscale['percent']
            del kwargspar['percent']
        elif 'sigma' in kwargsscale:
            kwargsscale['sigma'] = kwargsscale['sigma']/kwargsscale['central']
            del kwargspar['sigma']
        else:
            raise Exception('Invalid parameter configuration')

        kwargsscale['central'] = 1.0
        kwargspar['fixed'] = True

        return sepuncfmt.format(name), kwargsscale

    def load_cfg(self):
        pars = cfg_parse(self.cfg.pars, verbose=True)
        action = self.cfg.get('hooks')
        if isinstance(action, (Mapping, NestedDict)):
            for k, act in action.items():
                v, l = act(pars)
                pars[k]=v
                if l:
                    pars.setdefault('meta',{}).setdefault('labels', {})[k]=l

        pars = DictWrapper(pars, split='.')
        return pars

    def define_variables(self):
        self._par_container = []
        pars = self.load_cfg()
        labels = self.cfg.get('labels', pars.get(('meta', 'labels'), {}))
        objectize = self.cfg.get('objectize')
        skip = list(self.cfg.get('skip', ()))+self.skip

        state = self.cfg.get('state', 'fixed')
        uncertainty_common = pars.get('uncertainty', None)
        uncertainty_mode_common = pars.get('uncertainty_mode', 'absolute')
        for name in pars.keys():
            if name in skip:
                continue

            for it_major in self.nidx_major:
                major_values = it_major.current_values()
                kwargspar=self.get_parameter_kwargs(pars, name, major_values, state=state, uncertainty_common=uncertainty_common, uncertainty_mode_common=uncertainty_mode_common)

                if name in labels:
                    kwargspar['label'] = labels[name]

                scalename, kwargsscale = self.get_scale_kwargs(name, kwargspar)

                for it_minor in self.nidx_minor:
                    it=it_major+it_minor

                    if kwargsscale:
                        scalepar = self.reqparameter(scalename, it, **kwargsscale)

                    par = self.reqparameter(name, it, **kwargspar)

                    if objectize:
                        trans=par.transformations.value
                        trans.setLabel(kwargspar.get('label', ''))
                        self.set_output(name, it, trans.single())

                        self._par_container.append(par)

                        if kwargsscale:
                            trans=scalepar.transformations.value
                            trans.setLabel(kwargsscale.get('label', ''))
                            self.set_output(scalename, it, trans.single())

                            self._par_container.append(scalepar)

    def build(self):
        pass
