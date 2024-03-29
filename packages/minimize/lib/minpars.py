import ROOT
import numpy as np

class MinPar(object):
    def __init__(self, base, par, *, initial_central: bool=True, **kwargs):
        assert base is not None

        self._base = base
        self._par  = par

        self.setup(initial_central=initial_central, **kwargs)

    def __str__(self):
        return '{name:<25} {_value:8}, limits=[{_vmin}, {_vmax}]' \
               '  constrained={_constrained}' \
               '  fixed={_fixed}' \
               '  step={_step}' \
               ''.format(**self.__dict__)

    def setup(self, *, initial_central: bool=True, **kwargs):
        self.name  = self._par.qualifiedName()
        self.fixed = kwargs.pop('fixed', self._par.isFixed())
        limits = self._par.limits()
        if len(limits) == 1:
            left, right = limits[0]
        elif len(limits) > 1:
            raise Exception("More borders than needed")
        else:
            left, right = None, None
        self.vmin = kwargs.pop('vmin', left)
        self.vmax = kwargs.pop('vmax', right)
        self.constrained = kwargs.pop('constrained', False)
        self.scanvalues = kwargs.pop('scanvalues', None)

        value = kwargs.pop('value', None)
        if value is None:
            if initial_central:
                self.value = self._par.central()
            else:
                self.value = self._par.value()
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
    _initial_central = True
    def __init__(self, pars, check, *, initial_central: bool=True):
        self._specs={}
        self._parmap={}
        self._modified=True
        self._resized=True
        self._initial_central = initial_central

        self._skippars=[]
        for _, v in pars.items():
            if v.influences(check):
                self.addpar(v)
            else:
                self._skippars.append(v)

    def __str__(self):
        return str(self._specs)

    def dump(self):
        for i, (_, v) in enumerate(self._specs.items()):
            print('% 3d'%i, v)

    def parspec(self, idx):
        if isinstance(idx, str):
            return self._specs[idx]
        elif isinstance(idx, int):
            return list(self._specs.values())[idx]

        # Assume idx is parameter instance
        return self._parmap[idx]

    __getitem__ = parspec

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __len__(self):
        return len(self._specs)

    def names(self):
        return self._specs.keys()

    def specs(self):
        return self._specs.values()

    def values(self):
        return [par.value for par in self._specs.values()]

    def items(self):
        return self._specs.items()

    def npars(self):
        return len(self._specs)

    def fixed(self):
        return [spec.name for spec in self._specs.values() if spec.fixed]

    def nfixed(self):
        return len(self.fixed())

    def nconstrained(self):
        return sum(1 for spec in self._specs.values() if spec.constrained and not spec.fixed)

    def nfree(self):
        return sum(1 for spec in self._specs.values() if not spec.constrained and not spec.fixed)

    def nvariable(self):
        return sum(1 for spec in self._specs.values() if not spec.fixed)

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

        spec = self._specs[name] = MinPar(self, par, initial_central=self._initial_central, **kwargs)
        self._parmap[par] = spec

        self.modified=True
        self.resized=True

    def pushpars(self):
        for par in self._parmap:
            par.push()

    def poppars(self):
        for par in self._parmap:
            par.pop()

    def __enter__(self):
        self.pushpars()

    def __exit__(self, exc_type, exc_value, traceback):
        self.poppars()

def par_influences(parameter, observables):
    for observable in observables:
        if parameter.influences(observable):
            return True
    return False

def partition(pred, iterable):
    trues = []
    falses = []
    for item in iterable:
        if pred(item):
            trues.append(item)
        else:
            falses.append(item)
    return trues, falses
