# -*- coding: utf-8 -*-
from __future__ import print_function
from gna.configurator import uncertain

debug=False

class DiscreteParameter(object):
    def __init__(self, name, variants, **kwargs):
        from gna import constructors as C
        self.default = None
        self._variable = C.ParameterWrapper(name)
        self._name = name
        self._namespace = kwargs.get("namespace", "")
        self._variants = variants
        self._inverse = dict(zip(variants.itervalues(), variants.iterkeys()))
        self._label = kwargs.pop('label', '')
        if  len(self._inverse) != len(self._variants):
            msg = "DiscreteParameter variants dict is not a bijection"
            raise Exception(msg)

        self.stack = []

    def name(self):
        return self._name

    def value(self):
        raw = self._variable.value()
        return self._inverse[raw]

    def set(self, val):
        self._variable.set(self._variants[val])

    def cast(self, val):
        return val

    def reset(self):
        if self.default is None:
            return
        self.set(self.default)

    def getVariable(self):
        return self._variable.getVariable()

    def setNamespace(self, name):
        self._namespace = name

    def getVariants(self):
        return tuple(self._variants)

    def getLabel(self):
        return self._label

    def setLabel(self, label):
        self._label=label

    def push(self, value=None):
        prev = self.value()
        if value is not None:
            self.stack.append(prev)
            self.set(value)
            return value

        self.stack.append(prev)
        return prev

    def pop(self):
        if self.stack:
            val = self.stack.pop()
            self.set(val)
            return val

        return self.value()

def makeparameter(ns, name, cfg=None, **kwargs):
    from gna import constructors as C
    if 'target' in kwargs:
        return kwargs['target']
    if cfg:
        ptype = kwargs['ptype'] = 'gaussian'
        kwargs['central']     = cfg.central
        if cfg.mode=='fixed':
            kwargs['uncertainty'] = 0.1
            kwargs['uncertainty_type'] = 'absolute'
            kwargs['fixed'] = True
        elif cfg.mode=='free':
            kwargs['uncertainty'] = float('inf')
            kwargs['uncertainty_type'] = 'absolute'
            kwargs['free'] = True
        else:
            kwargs['uncertainty'] = cfg.uncertainty
            kwargs['uncertainty_type'] = cfg.mode
        kwargs.setdefault('label', cfg.label)
    else:
        ptype = kwargs.get('type', 'gaussian')

    fixed = kwargs.get('fixed', False)
    free  = kwargs.get('free',  False)
    if free and fixed:
        raise Exception('Parameter {} may not be free and fixed in the same time')

    if fixed:
        if not 'relsigma' in kwargs:
            kwargs.setdefault('sigma', 1.e-6)
    elif free:
        if not 'relsigma' in kwargs and not 'sigma' in kwargs:
            kwargs['sigma']=float('inf')
            if not 'step' in kwargs:
                central = float(kwargs['central'])
                if central:
                    kwargs.setdefault('step', 0.1*central)
                else:
                    kwargs.setdefault('step', 0.1)

    if debug:
        print( 'Defpar {ns}.{name} ({type}):'.format(
            ns=ns.name, name=name, type=ptype
            ), end=' ' )
    if ptype == 'gaussian':
        param = C.GaussianParameter(name)
        if 'limits' in kwargs:
            upper, lower = kwargs['limits']
            param.addLimits(param.cast(upper), param.cast(lower))
            if debug:
                print( '({upper}, {lower})'.format(lower=lower, upper=upper), end=' ' )
        if 'central' in kwargs:
            param.setCentral(param.cast(kwargs['central']))
            if debug:
                print( kwargs['central'], end='' )
        else:
            msg = "parameter `%s': no central value" % name
            raise Exception(msg)

        if 'uncertainty' in kwargs and 'uncertainty_type' in kwargs:
            uncertainty, uncertainty_type = kwargs['uncertainty'], kwargs['uncertainty_type']

            unctypes = dict( relative='relsigma', absolute='sigma' )
            unckey = unctypes.get( uncertainty_type )
            if not unckey:
                raise Exception( "Unknown uncertainty type '%s'"%uncertainty_type )
            kwargs[unckey] = uncertainty

        if 'relsigma' in kwargs:
            rs = float(kwargs['relsigma'])
            if rs==0.0:
                fixed = True

            sigma = param.central()*rs
            if 'sigma' in kwargs and sigma != kwargs['sigma']:
                msg = ("parameter `%s': conflicting relative (%g*%g=%g)"
                       "and absolute (%g) sigma values")
                msg = msg % (name, param.central(), rs, sigma,
                             kwargs['sigma'])
                raise Exception(msg)
            param.setSigma(sigma)
            if debug:
                print( u'*(1±{relsigma}) [±{sigma}] [{perc}%]'.format(sigma=sigma,relsigma=rs,perc=rs*100.0), end=' ' )
        elif 'sigma' in kwargs:
            sigma = param.cast(kwargs['sigma'])
            if sigma==0.0:
                fixed=True

            param.setSigma(sigma)
            if debug:
                print( u'±{sigma}'.format(sigma=param.sigma() ), end=' ' )
                if param.central():
                    print( '[{perc}%]'.format(perc=param.sigma()/param.central() ), end=' ' )
        else:
            msg = "parameter `%s': no sigma value" % name
            raise Exception(msg)
    elif ptype == 'discrete':
        if 'variants' not in kwargs:
            msg = "parameter `%s': no discrete variants" % name
            raise Exception(msg)
        param = DiscreteParameter(name, kwargs['variants'])
        if 'default' in kwargs:
            param.default = kwargs['default']
        if debug:
            print( '{default} {variants}'.format( default=kwargs['variants'][kwargs['default']], variants=kwargs['variants'] ), end=' ' )
    elif ptype == 'uniformangle':
        param = C.UniformAngleParameter(name)
        if 'central' in kwargs:
            param.setCentral(param.cast(kwargs['central']))
        else:
            raise Exception( "parameter `%s': no central value" % name)

        if debug:
            print( '{central} rad'.format( central=param.central() ), end=' ' )

    if fixed:
        param.setFixed()
        if debug:
            print( 'fixed!', end='' )
    elif free:
        param.setFree()
        if debug:
            print( 'free!', end='' )
    if 'step' in kwargs:
        param.setStep(kwargs['step'])
        if debug:
            print( 'step={}'.format(kwargs['step']), end='' )

    param.reset()
    param.ns = ns

    if 'label' in kwargs:
        param.setLabel(kwargs['label'])
    if debug:
        print()
    param.setNamespace(ns.path)
    return param

