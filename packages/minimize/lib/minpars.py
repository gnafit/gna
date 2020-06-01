from __future__ import print_function
import ROOT
import numpy as np
from collections import OrderedDict

class MinPar(object):
    def __init__(self, base, par, **kwargs):
        assert base

        self._base = base
        self._par  = par

        self.setup(**kwargs)

    def __str__(self):
        return '{name:<25} {_value:8}, limits=[{_vmin}, {_vmax}]' \
               '  constrained={_constrained}' \
               '  fixed={_fixed}' \
               '  step={_step}' \
               ''.format(**self.__dict__)

    def setup(self, **kwargs):
        self.name  = self._par.qualifiedName()
        self.fixed = kwargs.pop('fixed', self._par.isFixed())
        self.vmin = kwargs.pop('vmin', None)
        self.vmax = kwargs.pop('vmax', None)
        self.constrained = kwargs.pop('constrained', False)
        self.scanvalues = kwargs.pop('scanvalues', None)

        value = kwargs.pop('value', None)
        if value is None:
            self.value = self._par.central()
        else:
            self.value = value

        step = kwargs.pop('step', None)
        if step is None:
            self.step = self._par.step()
        else:
            self.step = step

        if step==0.0:
            raise Exception('"%s" initial step is undefined. Specify its sigma explicitly.'%self._par.qualifiedName())

        assert not kwargs, 'Unparsed MinPar arguments: {!s}'.format(kwargs)

    @property
    def par(self):
        return self._par

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self._base.modified = True

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, step):
        self._step = step
        self._base.modified = True

    @property
    def scanvalues(self):
        return self._scanvalues

    @scanvalues.setter
    def scanvalues(self, scanvalues):
        self._scanvalues = scanvalues
        self._base.modified = True

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, value):
        self._fixed = value
        self._base.modified = True

    @property
    def vmin(self):
        return self._vmin

    @vmin.setter
    def vmin(self, vmin):
        self._vmin = vmin
        self._base.modified = True

    @property
    def vmax(self):
        return self._vmax

    @vmax.setter
    def vmax(self, vmax):
        self._vmax = vmax
        self._base.modified = True

    @property
    def constrained(self):
        return self._constrained

    @constrained.setter
    def constrained(self, constrained):
        self._constrained = constrained
        # No need to modify the base as it does not affect the minimization behaviour (it is just a flag)

class MinPars(object):
    def __init__(self, pars):
        self._specs=OrderedDict()
        self._parmap=OrderedDict()
        self._modified=True
        self._resized=True

        for k, v in pars.items():
            self.addpar(v)

    def dump(self):
        for i, (k, v) in enumerate(self._specs.items()):
            print('% 3d'%i, v)

    def parspec(self, idx):
        if isinstance(idx, str):
            return self._specs[idx]
        elif isinstance(idx, int):
            return list(self._specs.values())[idx]

        # Assume idx is parameter instance
        return self._parmap[idx]

    def names(self):
        return self._specs.keys()

    def specs(self):
        return self._specs.values()

    def items(self):
        return self._specs.items()

    def npars(self):
        return len(self._specs)

    def nfixed(self):
        return sum(1 for spec in self._specs if spec.fixed)

    def nconstrained(self):
        return sum(1 for spec in self._specs if spec.constrained and not spec.fixed)

    def nfree(self):
        return sum(1 for spec in self._specs if not spec.constrained and not spec.fixed)

    def resetstatus(self):
        self.modified = False
        self.resized = False

    @property
    def modified(self):
        return self._modified

    @modified.setter
    def modified(self, modified):
        self._modified = modified

    @property
    def resized(self):
        return self._resized

    @resized.setter
    def resized(self, resized):
        self._resized = resized

    def addpar(self, par, **kwargs):
        name = par.qualifiedName()
        if name in self._specs or par in self._specs.values():
            raise Exception('The parameter {} added twice'.format(name))

        spec = self._specs[name] = MinPar(self, par, **kwargs)
        self._parmap[par] = spec

        self.modified=True
        self.resized=True

    def pushpars(self):
        for par in self._parmap:
            par.push()

    def poppars(self):
        for par in self._parmap:
            par.pop()
