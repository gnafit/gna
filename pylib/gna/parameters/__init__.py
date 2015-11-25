import ROOT

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
    if ptype == 'gaussian':
        param = ROOT.GaussianParameter("double")(name)
        if 'limits' in kwargs:
            upper, lower = kwargs['limits']
            param.addLimits(param.cast(upper), param.cast(lower))
        if 'central' in kwargs:
            param.setCentral(param.cast(kwargs['central']))
        else:
            msg = "parameter `%s': no central value" % name
            raise Exception(msg)
        if 'relsigma' in kwargs:
            rs = kwargs['relsigma']
            sigma = param.central()*rs
            if 'sigma' in kwargs and sigma != kwargs['sigma']:
                msg = ("parameter `%s': conflicting relative (%g*%g=%g)"
                       "and absolute (%g) sigma values")
                msg = msg % (name, param.central(), rs, sigma,
                             kwargs['sigma'])
                raise Exception(msg)
            param.setSigma(sigma)
        elif 'sigma' in kwargs:
            param.setSigma(kwargs['sigma'])
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
    elif ptype == 'uniformangle':
        param = ROOT.UniformAngleParameter("double")(name)
        if 'central' in kwargs:
            param.setCentral(param.cast(kwargs['central']))
        else:
            raise Exception( "parameter `%s': no central value" % name)
    param.reset()
    param.ns = ns
    return param
