from collections import defaultdict, Counter
from gna import parameters
from contextlib import contextmanager
import ROOT

class namespace(object):
    def __init__(self, env):
        self.vars = []
        self.namespaces = defaultdict(lambda: namespace(env))
        self.parameters = {}
        self.evaluables = {}
        self.expressions = {}
        self.observables = {}
        self.env = env

    def __call__(self, *args):
        return self.ns(*args)

    def __getitem__(self, name):
        for container in (self.parameters, self.evaluables):
            try:
                return container[name]
            except KeyError:
                pass

    def __contains__(self, name):
        return name in self.parameters or name in self.evaluables

    def ns(self, parts):
        if isinstance(parts, str):
            return self.ns(parts.split('.'))
        if len(parts) > 0:
            return self.namespaces[parts[0]].ns(parts[1:])
        else:
            return self

    def defparameter(self, name, **kwargs):
        return self.env.defparameter(self, name, **kwargs)

    def getparameter(self, name):
        return self.parameters[name]

    def getevaluable(self, name):
        return self.evaluables[name]

    def getexpressions(self, name):
        return self.expressions[name]

    def names(self):
        return set(self.parameters.keys()).union(self.evaluables.keys())

    def __enter__(self):
        self.env.nsview.ref([self])

    def __exit__(self, type, value, tb):
        self.env.nsview.deref([self])

    def addobservable(self, name, output):
        self.observables[name] = output

    def iternstree(self, nspath=None):
        if nspath:
            yield (nspath, self)
            subprefix = nspath+'.'
        else:
            yield ('', self)
            subprefix = ''
        for name, subns in self.namespaces.iteritems():
            for x in subns.iternstree(nspath=subprefix+name):
                yield x

def foreachns(f):
    def wrapped(self, name):
        for ns, refs in self.nses.iteritems():
            if not refs > 0:
                continue
            try:
                return f(ns, name)
            except KeyError:
                pass
        raise KeyError(name)
    return wrapped

class nsview(object):
    def __init__(self):
        self.nses = Counter()

    def ref(self, nses):
        self.nses.update(nses)

    def deref(self, nses):
        self.nses.subtract(nses)

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
        return self.env.nsview.getparameter(name)

    @contextmanager
    def update(self, newvalues):
        oldvalues = {}
        for p, v in newvalues.iteritems():
            if isinstance(p, str):
                p = self[p]
            oldvalues[p] = p.value()
            p.set(v)
        yield
        for p, v in oldvalues.iteritems():
            p.set(v)

class observablesview(object):
    def __init__(self, env):
        self.env = env

    def iterpaths(self):
        for nspath, ns in self.env.iternstree():
            for name, prediction in ns.observables.iteritems():
                yield '/'.join([nspath, name])

    def fromspec(self, spec):
        ret = []
        for path in spec.split('+'):
            nspath, name = path.split('/')
            ret.append(self.env.ns(nspath).observables[name])
        return ret

class _environment(object):
    def __init__(self):
        self.objs = []
        self.parameters = parametersview(self)
        self.pars = self.parameters
        self.observables = observablesview(self)
        self.predictions = {}
        self.covmats = {}
        self.data = {}
        self.globalns = namespace(self)
        self.nsview = nsview()
        self.nsview.ref([self.globalns])
        self._bindings = []

    def setns(self, ns):
        self.currentns = ns
        self.currentview = nsview([ns, self.globalns])

    def view(self, ns):
        if ns != self.globalns:
            return nsview([ns, self.globalns])
        else:
            return nsview([self.globalns])

    def register(self, obj, **kwargs):
        self.objs.append(obj)
        ns = kwargs.pop('ns', self.globalns)
        bindings = self._bindings+[kwargs.pop("bindings", {})]
        if len(obj.evaluables):
            self.addexpressions(obj, ns=ns, bindings=bindings)
        if not kwargs.pop('bind', True):
            return obj
        if isinstance(obj, ROOT.ExpressionsProvider):
            return obj
        resolver = parameters.resolver(self, self.nsview)
        resolver.resolveobject(obj, bindings=bindings,
                               freevars=kwargs.pop('freevars', []))
        return obj

    def ns(self, ns):
        if isinstance(ns, namespace):
            return ns
        elif isinstance(ns, str):
            return self.globalns.ns(ns)
        else:
            raise Exception("unknown object %r passed to ns()" % ns)

    def iternstree(self):
        return self.globalns.iternstree(nspath='')

    @contextmanager
    def bind(self, **bindings):
        self._bindings.append(bindings)
        yield
        self._bindings.pop()

    def defparameter(self, *args, **kwargs):
        if len(args) == 1:
            ns, name = self.globalns, args[0]
        elif len(args) == 2:
            ns, name = args
        else:
            raise TypeError("defparameter() takes 1 or 2 arguments"
                            "({0} given)".format(len(args)))
        assert name not in ns.parameters
        param = parameters.makeparameter(name, **kwargs)
        ns.parameters[name] = param
        return param

    def addevaluable(self, ns, name, var):
        evaluable = ROOT.GaussianValue(var.typeName())(name, var)
        ns.evaluables[name] = evaluable
        return evaluable

    def addexpression(self, obj, expr, ns, bindings=[]):
        deps = [next((bs[name] for bs in bindings if name in bs), name)
                for name in expr.sources]
        entry = (obj, expr, deps, ns)
        ns.expressions.setdefault(expr.name, []).append(entry)

    def addexpressions(self, obj, ns, bindings=[]):
        for expr in obj.evaluables.itervalues():
            self.addexpression(obj, expr, ns, bindings=bindings)

    def addprediction(self, name, prediction):
        self.predictions[name] = prediction

    def addcovmat(self, name, covmat):
        self.covmats[name] = covmat

    def adddata(self, name, data):
        self.data[name] = data

    def gettype(self, objtype):
        types = {
            'prediction': 'predictions',
            'observable': 'observables',
            'data': 'data',
            'covmat': 'covmats',
            'parameter': 'parameters',
        }
        matches = [k for k in types if k.startswith(objtype)]
        if len(matches) > 1:
            msg = "ambigous type specifier {0}, candidates: {1}"
            raise Exception(msg.format(objtype, ', '.join(matches)))
        elif not matches:
            msg = "unknown type specifier {0}"
            raise Exception(msg.format(objtype))
        else:
            return types[matches[0]]

    def get(self, objspec):
        objtype, objpath = objspec.split(":", 1)
        objtype = self.gettype(objtype)
        if '/' in objpath:
            nspath, objname = objpath.rsplit("/", 1)
            return getattr(self.ns(nspath), objtype)[objname]
        else:
            return getattr(self, objtype)[objpath]

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
