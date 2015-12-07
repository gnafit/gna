from collections import defaultdict, deque, Mapping
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

class entrydict(dict):
    def __init__(self, ns):
        self.ns = ns

class namespace(Mapping):
    def __init__(self, env, parent, name):
        self.env = env

        self.name = name
        self.parent = parent
        if parent is not None:
            self.path = '.'.join([parent.path, name])
        else:
            self.path = name

        self.storage = defaultdict(lambda: entrydict(self))
        self.rules = []
        self.namespaces = namespacedict(env, self)
        self.observables = {}

        self.objs = []

    def __nonzero__(self):
        return True

    def __repr__(self):
        return "<namespace path='{0}'>".format(self.path)

    def __enter__(self):
        self.env.nsview.add([self])

    def __exit__(self, type, value, tb):
        self.env.nsview.remove([self])

    def __call__(self, nsname):
        if isinstance(nsname, basestring):
            parts = nsname.split('.')
        else:
            parts = nsname
        if not parts:
            return self
        return self.namespaces[parts[0]](parts[1:])

    def __getitem__(self, name):
        entry = self.storage[name]
        for t in ('alias', 'parameter', 'evaluable'):
            if t in entry:
                return entry[t]
        else:
            raise KeyError(name)

    def __iter__(self):
        return self.storage.iterkeys()

    def __len__(self):
        return len(self.storage)

    def getentry(self, name):
        if name not in self.storage:
            raise KeyError(name)
        return self.storage[name]

    def defparameter(self, name, **kwargs):
        target = self.matchrule(name)
        entry = self.storage[name]
        if not target:
            target = kwargs.pop('target', None)
        if target:
            entry['alias'] = target
        else:
            param = parameters.makeparameter(self, name, **kwargs)
            entry['parameter'] = param

    def addobservable(self, name, output):
        if output.check():
            self.observables[name] = output
        else:
            print "observation", name, "is invalid"
            output.dump()

    def addexpression(self, obj, expr, bindings=[]):
        deps = [next((bs[name] for bs in bindings if name in bs), name)
                for name in expr.sources]
        entry = (obj, expr, deps)
        self.storage[expr.name].setdefault('expressions', []).append(entry)

    def addevaluable(self, name, var):
        evaluable = ROOT.GaussianValue(var.typeName())(name, var)
        evaluable.ns = self
        self.storage[name]['evaluable'] = evaluable
        return evaluable

    def iternstree(self):
        yield self
        for name, subns in self.namespaces.iteritems():
            for x in subns.iternstree():
                yield x

    def ref(self, name):
        return '.'.join([self.path, name])

    def matchrule(self, name):
        for pattern, target in self.rules:
            if not pattern or pattern(name):
                return target

class nsview(object):
    def __init__(self):
        self.nses = deque()

    def add(self, nses):
        self.nses.extendleft(nses)

    def remove(self, nses):
        for ns in nses:
            self.nses.remove(ns)

    def getentry(self, name):
        for ns in self.nses:
            try:
                return ns.getentry(name)
            except KeyError:
                pass
        raise KeyError(name)

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
            return self.env.nsview.getentry(name)['parameter']

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

    @contextmanager
    def save(self, params):
        oldvalues = {}
        for p in params:
            if isinstance(p, str):
                p = self[p]
            oldvalues[p] = p.value()
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
        for expr in obj.evaluables.itervalues():
            ns.addexpression(obj, expr, bindings=bindings)
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
            return self.globalns(ns)
        else:
            raise Exception("unknown object %r passed to ns()" % ns)

    def defparameter(self, name, **kwargs):
        if '.' in name:
            nsname, name = name.rsplit('.', 1)
            return self.ns(nsname).defparameter(name, **kwargs)
        else:
            return self.globalns.defparameter(name, **kwargs)

    def iternstree(self):
        return self.globalns.iternstree()

    def bind(self, **bindings):
        self._bindings.append(bindings)
        yield
        self._bindings.pop()

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
