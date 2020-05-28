from __future__ import print_function
import ROOT
import numpy as np
from collections import OrderedDict

class MinPar(object):
    def init(self, base, par, **kwargs):
        assert base

        self._base = base
        self._par  = par

        self.fixed = kwargs.pop('fixed', par.isFixed())
        self.limits = kwargs.pop('limits', False)
        self.constrained = kwargs.pop('constrained', False)
        self.scanvalues = kwargs.pop('scanvalues', None)

        value = kwargs.pop('value', None)
        if value is None:
            self.value = par.central()
        else:
            self.value = value

        step = kwargs.pop('step', None)
        if step is None:
            self.step = par.step()
        else:
            self.step = step

        if step==0.0
            raise Exception('"%s" initial step is undefined. Specify its sigma explicitly.'%par.qualifiedName())

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
    def limits(self):
        return self._limits

    @limits.setter
    def limits(self, limits):
        self._limits = limits
        self._base.modified = True

    @property
    def constrained(self):
        return self._constrained

    @constrained.setter
    def constrained(self, constrained):
        self._constrained = constrained
        # No need to modify the base as it does not affect the minimization behaviour (it is just a flag)

class MinPars(object):
    def __init__(self):
        self.pars=OrderedDict()
        self._modified=True

    @property
    def modified(self):
        return self._modified

    @modified.setter
    def modified(self, modified):
        self._modified = modified

    def addpar(self, par, **kwargs):
        name = par.qualifiedName()
        if name in self.pars or par in self.pars.values():
            raise Exception('The parameter {} added twice'.format(name))

