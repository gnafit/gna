from collections import defaultdict, deque
import parameters
import parresolver
from contextlib import contextmanager
import ROOT

class namespacedict(defaultdict):
    def __init__(self, env, ns):
        super(namespacedict, self).__init__()
        self.env = env
        self.ns = ns

    def __missing__(self, key):
        value = namespace(self.env, self.ns, key)
        self[key] = value
        return value

class namespace(object):
    def __init__(self, env, parent, name):
        self.objs = []
        self.namespaces = namespacedict(env, self)

        self.aliases = {}
        self.parameters = {}
        self.evaluables = {}

        self.expressions = {}

        self.observables = {}
        self.env = env
        self.parent = parent
        self.name = name

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

    def defalias(self, name, target):
        assert name not in self.aliases
        self.aliases[name] = target
        return target

    def defparameter(self, name, **kwargs):
        assert name not in self.parameters
        param = parameters.makeparameter(self, name, **kwargs)
        self.parameters[name] = param
        return param

    def getalias(self, name):
        return self.aliases[name]

    def getparameter(self, name):
        return self.parameters[name]

    def getevaluable(self, name):
        return self.evaluables[name]

    def getexpressions(self, name):
        return self.expressions[name]

    def names(self):
        return set(self.parameters.keys()).union(self.evaluables.keys())

    def __enter__(self):
        self.env.nsview.add([self])

    def __exit__(self, type, value, tb):
        self.env.nsview.remove([self])

    def addobservable(self, name, output):
        if output.check():
            self.observables[name] = output
        else:
            print "observation", name, "is invalid"
            output.dump()

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

    def path(self):
        if not self.parent:
            return self.name
        return '.'.join([self.parent.path(), self.name])

    def ref(self, name):
        return '.'.join([self.path(), name])

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
    def __init__(self):
        self.nses = deque()

    def add(self, nses):
        self.nses.extendleft(nses)

    def remove(self, nses):
        for ns in nses:
            self.nses.remove(ns)

    @foreachns
    def getalias(ns, name):
        return ns.getalias(name)

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
        if '.' in name:
            nspath, name = name.rsplit('.', 1)
            return self.env.ns(nspath)[name]
        else:
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

class PartNotFoundError(BaseException):
    def __init__(self, parttype, partname):
        self.parttype = parttype
        self.partname = partname

class envpart(dict):
    def __init__(self, parttype):
        self.parttype = parttype
        super(envpart, self).__init__()

    def __hash__(self):
        return hash(self.parttype)

    def __call__(self, name):
        try:
            return self[name]
        except KeyError:
            raise PartNotFoundError(self.parttype, name)

class envparts(object):
    def __init__(self):
        self.storage = {}

    def __getattr__(self, parttype):
        if not parttype in self.storage:
            self.storage[parttype] = envpart(parttype)
        return self.storage[parttype]

class _environment(object):
    def __init__(self):
        self._bindings = []

        self.globalns = namespace(self, None, '')
        self.nsview = nsview()
        self.nsview.add([self.globalns])

        self.parameters = parametersview(self)
        self.pars = self.parameters
        self.parts = envparts()

    def view(self, ns):
        if ns != self.globalns:
            return nsview([ns, self.globalns])
        else:
            return nsview([self.globalns])

    def register(self, obj, **kwargs):
        ns = kwargs.pop('ns')
        if not ns:
            ns = self.globalns
        ns.objs.append(obj)
        bindings = self._bindings+[kwargs.pop("bindings", {})]
        if len(obj.evaluables):
            self.addexpressions(obj, ns=ns, bindings=bindings)
        if not kwargs.pop('bind', True):
            return obj
        if isinstance(obj, ROOT.ExpressionsProvider):
            return obj
        resolver = parresolver.resolver(self, self.nsview)
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

    def addevaluable(self, ns, name, var):
        evaluable = ROOT.GaussianValue(var.typeName())(name, var)
        evaluable.ns = ns
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

    def gettype(self, objtype):
        types = self.parts.storage
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
        return self.gettype(objtype)[objpath]

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
