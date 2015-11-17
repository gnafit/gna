import yaml
import numpy as np

class MinimizerSpecError(Exception):
    pass

def loadlimits(par, limits):
    if not isinstance(limits, list) or len(limits) != 2:
        msg = "invalid limits specifier {0}, should be [min, max]"
        raise MinimizerSpecError(msg.format(limits))
    try:
        limits = [float(v) for v in limits]
    except ValueError:
        msg = "invalid limit value {0}, should be float"
        raise MinimizerSpecError(msg.format(v))
    if limits[1] <= limits[0]:
        msg = "invalid limit values {0}, min should be less than max"
        raise MinimizerSpecError(msg.format(limits))
    return limits

class dynamicvalue(object):
    pass

class central(dynamicvalue):
    @classmethod
    def construct(cls, loader, node):
        if not isinstance(node, yaml.ScalarNode) or node.value != '':
            msg = "{0} has no arguments"
            raise MinimizerSpecError(msg.format(node.tag))
        return cls()

    def value(self, par):
        return par.central()

yaml.add_constructor('!central', central.construct)

def loadvalue(par, value):
    if isinstance(value, central):
        return value
    try:
        return par.cast(value)
    except ValueError:
        msg = "invalid value {0} for parameter {1}"
        raise MinimizerSpecError(msg.format(value, par.name()))

def loadstep(par, step):
    try:
        return float(step)
    except ValueError:
        msg = "invalid step value {0}, should be float"
        raise MinimizerSpecError(msg.format(step))

def loadfixed(par, fixed):
    if not isinstance(fixed, bool):
        msg = "invalid fixed {0}, should be bool"
        raise MinimizerSpecError(msg.format(fixed))
    return fixed

fieldfuncs = {
    'limits': loadlimits,
    'value': loadvalue,
    'step': loadstep,
    'fixed': loadfixed,
}

def loadfields(par, fields):
    ret = {}
    for field, value in fields.iteritems():
        try:
            func = fieldfuncs[field]
        except KeyError:
            msg = "undefined field {0}"
            raise MinimizerSpecError(msg.format(field))
        if value is not None:
            ret[field] = func(par, value)
        else:
            ret[field] = None
    return ret

def load(env, minimizer, spec):
    ret = {}
    for k, fields in spec.iteritems():
        par = None
        if isinstance(k, basestring):
            try:
                par = env.pars[k]
            except KeyError:
                msg = "undefined parameter {0}"
                raise MinimizerSpecError(msg.format(k))
            try:
                minimizer.pars.index(par)
            except ValueError:
                msg = "no parameter {0} in minimizer"
                raise MinimizerSpecError(par.name())
        elif isinstance(k, int):
            try:
                par = minimizer.pars[k]
            except IndexError:
                msg = "invalid index {0}, have {1} parameters in minimizer"
                raise MinimizerSpecError(msg.format(k, len(minimizer.pars)))
        else:
            msg = "invalid parameter key {0}, except index or name"
            raise MinimizerSpecError(msg.format(k))

        ret[par] = loadfields(par, fields)
    return ret

def parse(env, minimizer, s):
    spec = yaml.load(s)
    if spec is None:
        spec = {}
    return load(env, minimizer, spec)

def merge(basespec, spec):
    merged = {}
    for k in spec:
        if spec[k] is None:
            continue
        fields = dict(basespec.get(k, {}).items())
        fields.update(spec[k])
        merged[k] = {field: v for (field, v) in fields.iteritems() if v is not None}
    return merged
