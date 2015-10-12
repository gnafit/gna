from collections import defaultdict
from gna import parameters
import ROOT

class namespace(object):
    def __init__(self, env):
        self.objs = []
        self.vars = []
        self.namespaces = defaultdict(lambda: namespace(env))
        self.parameters = {}
        self.evaluables = {}
        self.expressions = defaultdict(list)
        self.env = env
        self.oldns = None

    def getns(self, parts):
        if len(parts) > 0:
            return self.namespaces[parts[0]].getns(parts[1:])
        else:
            return self

    def getparameter(self, name):
        return self.parameters[name]

    def getevaluable(self, name):
        return self.evaluables[name]

    def getexpressions(self, name):
        return self.expressions[name]

    def names(self):
        return set(self.parameters.keys()).union(self.evaluables.keys())

    def __enter__(self):
        self.oldns = self.env.currentns
        self.env.setns(self)

    def __exit__(self, type, value, tb):
        self.env.setns(self.oldns)
        self.oldns = None

def foreachns(f):
    def wrapped(self, name):
        for ns in self.nses:
            try:
                return f(ns, name)
            except KeyError:
                pass
        raise KeyError(name)
    return wrapped

class nsview(object):
    def __init__(self, nses):
        self.nses = nses

    @foreachns
    def getparameter(ns, name):
        return ns.getparameter(name)

    @foreachns
    def getevaluable(ns, name):
        return ns.getevaluable(name)

    @foreachns
    def getexpressions(ns, name):
        return ns.getexpressions(name)

    def names(self):
        return set.union(*[ns.names() for ns in self.nses])

class parametersview(object):
    def __init__(self, env):
        self.env = env

    def __getitem__(self, name):
        return self.env.currentview.getparameter(name)

class _environment(object):
    def __init__(self):
        self.pars = parametersview(self)
        self.globalns = namespace(self)
        self.setns(self.globalns)

    def setns(self, ns):
        self.currentns = ns
        self.currentview = nsview([ns, self.globalns])

    def view(self, ns):
        if ns != self.globalns:
            return nsview([ns, self.globalns])
        else:
            return nsview([self.globalns])

    def register(self, obj, **kwargs):
        ns = self.ns(kwargs.pop('ns', None))
        ns.objs.append(obj)
        if len(obj.evaluables):
            self.addexpressions(obj, bindings=kwargs.get("bindings", {}))
        if not kwargs.pop('bind', True):
            return obj
        if isinstance(obj, ROOT.ExpressionsProvider):
            return obj
        resolver = parameters.resolver(self, self.view(ns))
        resolver.resolveobject(obj, **kwargs)
        return obj

    def ns(self, ns=None):
        if not ns:
            return self.currentns
        if isinstance(ns, namespace):
            return ns
        parts = ns.split('.')
        if parts[0]:
            return self.currentns.getns(parts)
        else:
            return self.globalns.getns(parts)

    def defparameter(self, name, **kwargs):
        ns = self.ns()
        assert name not in ns.parameters
        param = parameters.makeparameter(name, **kwargs)
        ns.parameters[name] = param
        return param

    def addevaluable(self, name, var):
        evaluable = ROOT.GaussianValue("double")(name, var)
        self.ns().evaluables[name] = evaluable
        return evaluable

    def addexpression(self, obj, expr, bindings={}):
        deps = [bindings.get(name, name) for name in expr.sources]
        self.ns().expressions[expr.name].append((obj, expr, deps))

    def addexpressions(self, obj, bindings={}):
        for expr in obj.evaluables.itervalues():
            self.addexpression(obj, expr, bindings=bindings)

class _environments(defaultdict):
    current = None

    def __init__(self):
        super(_environments, self).__init__(_environment)

    def __getattr__(self, name):
        return self[name]

    def __getitem__(self, name):
        env = super(_environments, self).__getitem__(name)
        if _environments.current is None:
            _environments.current = env
        return env

env = _environments()
