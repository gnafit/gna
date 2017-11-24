# -*- coding: utf-8 -*-
from __future__ import print_function
import ROOT

debug=False

class DiscreteParameter(object):
    def __init__(self, name, variants):
        self.default = None
        self._variable = ROOT.ParameterWrapper("double")(name)
        self._name = name
        self._variants = variants
        self._inverse = dict(zip(variants.itervalues(), variants.iterkeys()))
        if  len(self._inverse) != len(self._variants):
            msg = "DiscreteParameter variants dict is not a bijection"
            raise Exception(msg)

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

def makeparameter(ns, name, **kwargs):
    if 'target' in kwargs:
        return kwargs['target']
    ptype = kwargs.get('type', 'gaussian')

    if debug:
        print( 'Defpar {ns}.{name} ({type}):'.format(
            ns=ns.name, name=name, type=ptype
            ), end=' ' )
    if ptype == 'gaussian':
        param = ROOT.GaussianParameter("double")(name)
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

            if not uncertainty_type in [ 'sigma', 'relsigma' ]:
                raise Exception( "Unknown uncertainty type '%s'"%uncertainty_type )
            kwargs[uncertainty_type] = uncertainty

        if 'relsigma' in kwargs:
            rs = float(kwargs['relsigma'])
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
            param.setSigma(param.cast(kwargs['sigma']))
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
        param = ROOT.UniformAngleParameter("double")(name)
        if 'central' in kwargs:
            param.setCentral(param.cast(kwargs['central']))
        else:
            raise Exception( "parameter `%s': no central value" % name)

        if debug:
            print( '{central} rad'.format( central=param.central() ), end=' ' )

    if 'fixed' in kwargs:
        param.setFixed()
        print( 'fixed!', end='' )
    param.reset()
    param.ns = ns

    if debug:
        print()
    return param
