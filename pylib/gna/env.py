from collections import defaultdict, deque, Mapping, OrderedDict
import parameters
from contextlib import contextmanager
import ROOT
from gna.config import cfg

env = None

class namespacedict(OrderedDict):
    def __init__(self, ns):
        super(namespacedict, self).__init__()
        self.ns = ns

    def __missing__(self, key):
        value = namespace(self.ns, key)
        self[key] = value
        return value

def findname(name, curns):
    if '.' in name:
        nsname, name = name.rsplit('.', 1)
        ns = env.ns(nsname)
    else:
        ns = curns
    return ns[name]

class ExpressionWithBindings(object):
    def __init__(self, ns, obj, expr, bindings):
        self.ns = ns
        self.obj = obj
        self.expr = expr
        self.bindings = bindings

    def resolvepath(self, seen, known):
        allpath = []
        for src in self.expr.sources.itervalues():
            depname = src.name()
            dep = next((bs[depname] for bs in self.bindings if depname in bs), depname)
            if isinstance(dep, basestring):
                if dep in known:
                    continue
                try:
                    dep = findname(dep, env.nsview)
                except KeyError:
                    return None
            if isinstance(dep, ExpressionsEntry):
                if dep in seen:
                    return None
                path = dep.resolvepath(seen | {dep}, known)
                if path is None:
                    return None
                known.append(dep)
                allpath.extend(path)
        return allpath

    def get(self):
        for src in self.expr.sources.itervalues():
            depname = src.name()
            v = self.obj.variables[depname]
            if not v.isFree():
                continue
            dep = next((bs[depname] for bs in self.bindings if depname in bs), depname)
            if isinstance(dep, basestring):
                dep = findname(dep, env.nsview)
            v.bind(dep.getVariable())
        return self.ns.addevaluable(self.expr.name(), self.expr.get())

class ExpressionsEntry(object):
    def __init__(self, ns):
        self.ns = ns
        self.exprs = []

    def add(self, obj, expr, bindings):
        self.exprs.append(ExpressionWithBindings(self.ns, obj, expr, bindings))

    def get(self):
        path = self.resolvepath({self}, OrderedDict())
        if not path:
            raise KeyError()
        for expr in path:
            v = expr.get()
        return v

    def resolvepath(self, seen, known):
        minexpr, minpaths = None, None
        for expr in self.exprs:
            if cfg.debug_bindings:
                print expr.expr.name(), seen
            paths = expr.resolvepath(seen, set(known))
            if paths is None:
                continue
            if len(paths) == 0:
                return [expr]
            if minpaths is None or len(minpaths) > len(paths):
                minexpr = expr
                minpaths = paths
        if minexpr is None:
            return None
        return minpaths+[minexpr]

class namespace(Mapping):
    def __init__(self, parent, name):
        self.name = name
        if parent is not None and parent.path:
            self.path = '.'.join([parent.path, name])
        else:
            self.path = name

        self.storage = OrderedDict()
        self.observables = OrderedDict()
        self.observables_tags = defaultdict(set)

        self.rules = []
        self.namespaces = namespacedict(self)

        self.objs = []

    def __nonzero__(self):
        return True

    def __repr__(self):
        return "<namespace path='{0}'>".format(self.path)

    def __enter__(self):
        env.nsview.add([self])

    def __exit__(self, type, value, tb):
        env.nsview.remove([self])

    def __call__(self, nsname):
        if isinstance(nsname, basestring):
            parts = nsname.split('.')
        else:
            parts = nsname
        if not parts:
            return self
        return self.namespaces[parts[0]](parts[1:])

    def link(self, nsname, newns):
        self.namespaces[nsname] = newns

    def inherit(self, otherns):
        for nsname in otherns.namespaces:
            if nsname not in self.namespaces:
                self.namespaces[nsname] = otherns.namespaces[nsname]

    def __getitem__(self, name):
        v = self.storage[name]
        if isinstance(v, basestring):
            return findname(v, env.nsview)
        return v

    def __setitem__(self, name, value):
        self.storage[name] = value

    def __iter__(self):
        return self.storage.iterkeys()

    def __len__(self):
        return len(self.storage)

    def defparameter_group(self, *args, **kwargs):
        import gna.parameters.covariance_helpers as ch
        pars = [self.defparameter(name, **ctor_args) for name, ctor_args in args]

        covmat_passed =  kwargs.get('covmat')
        if covmat_passed is not None:
            ch.covariate_pars(pars, covmat_passed)

        cov_from_cfg = kwargs.get('covmat_cfg')
        if cov_from_cfg is not None:
            ch.CovarianceHandler(cov_from_cfg, pars).covariate_pars()

        return pars
        
    def defparameter(self, name, **kwargs):
        if name in self.storage:
            raise Exception("{} is already defined".format(name))
        target = self.matchrule(name)
        if not target:
            target = kwargs.pop('target', None)
        if target:
            p = target
        else:
            p = parameters.makeparameter(self, name, **kwargs)
        self[name] = p
        return p

    def reqparameter(self, name, **kwargs):
        def without_status(name, **kwargs):
            try:
                par = self[name]
                return par
            except KeyError:
                pass
            try:
                par = env.nsview[name]
                return par
            except KeyError:
                pass
            return self.defparameter(name, **kwargs)

        def with_status(name, **kwargs):
            found = False
            try:
                par = self[name]
                found = True
                return par, found
            except KeyError:
                pass
            try:
                par = env.nsview[name]
                found = True
                return par, found 
            except KeyError:
                pass
            return self.defparameter(name, **kwargs), found

        if kwargs.get('with_status'):
            return with_status(name, **kwargs)
        else:
            return without_status(name, **kwargs)

    def reqparameter_group(self, *args, **kwargs):
        import gna.parameters.covariance_helpers as ch
        args_patched = [(name, dict(ctor_args, with_status=True))
                        for name, ctor_args in args]
        pars_with_status = [self.reqparameter(name, **ctor_args)
                            for name, ctor_args in args_patched]
        statuses = [status for _, status in pars_with_status]
        pars = [par for par, _ in pars_with_status]
        if not any(statuses):
            covmat_passed =  kwargs.get('covmat')
            if covmat_passed is not None:
                ch.covariate_pars(pars, covmat_passed)
            cov_from_cfg = kwargs.get('covmat_cfg')
            if cov_from_cfg is not None:
                ch.CovarianceHandler(cov_from_cfg, pars).covariate_pars()
        return pars

    def addobservable(self, name, output, export=True):
        if output.check():
            self.observables[name] = output
            print 'Add observable:', '%s/%s'%(self.path, name)
        else:
            print "observation", name, "is invalid"
            output.dump()
        if not export:
            self.observables_tags[name].add('internal')

    def addexpressions(self, obj, bindings=[]):
        for expr in obj.evaluables.itervalues():
            if cfg.debug_bindings:
                print self.path, obj, expr.name()
            name = expr.name()
            if name not in self.storage:
                self.storage[name] = ExpressionsEntry(self)
            if isinstance(self.storage[name], ExpressionsEntry):
                self.storage[name].add(obj, expr, bindings)

    def addevaluable(self, name, var):
        evaluable = ROOT.Uncertain(var.typeName())(name, var)
        evaluable.ns = self
        self[name] = evaluable
        return evaluable

    def walknstree(self):
        yield self
        for name, subns in self.namespaces.iteritems():
            for x in subns.walknstree():
                yield x

    def walkobservables(self, internal=False):
        for ns in self.walknstree():
            for name, val in ns.observables.iteritems():
                if not internal and 'internal' in ns.observables_tags.get(name, OrderedDict()):
                    continue
                yield '{}/{}'.format(ns.path, name), val

    def walknames(self):
        for ns in self.walknstree():
            for name, val in ns.storage.iteritems():
                yield '{}.{}'.format(ns.path, name), val

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

    def __getitem__(self, name):
        for ns in self.nses:
            try:
                return ns[name]
            except KeyError:
                pass
        if cfg.debug_bindings:
            print "can't find name {}. Names in view: ".format(name),
            if self.nses:
                for ns in self.nses:
                    print '"{}": "{}"'.format(ns.path, ', '.join(ns.storage)), ' ',
                print ''
            else:
                'none'
        raise KeyError('%s (namespaces: %s)'%(name, str([ns.name for ns in self.nses])))

class parametersview(object):
    def __getitem__(self, name):
        res = findname(name, env.nsview)
        return res

    @contextmanager
    def update(self, newvalues):
        oldvalues = OrderedDict()
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
        oldvalues = OrderedDict()
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
        self.storage = OrderedDict()

    def __getattr__(self, parttype):
        if not parttype in self.storage:
            self.storage[parttype] = envpart(parttype)
        return self.storage[parttype]

class _environment(object):
    def __init__(self):
        self._bindings = []

        self.globalns = namespace(None, '')
        self.nsview = nsview()
        self.nsview.add([self.globalns])

        self.parameters = parametersview()
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
            self.globalns.objs.append(obj)
        else:
            ns.objs.append(obj)
        bindings = self._bindings+[kwargs.pop("bindings", OrderedDict())]
        if ns:
            ns.addexpressions(obj, bindings=bindings)
        if not kwargs.pop('bind', True):
            return obj
        if isinstance(obj, ROOT.ExpressionsProvider):
            return obj
        freevars = kwargs.pop('freevars', [])

        for v in obj.variables.itervalues():
            if v.name() in freevars:
                continue
            if not v.isFree():
                continue
            vname = v.name()
            param = next((bs[vname] for bs in bindings if vname in bs), vname)
            if isinstance(param, basestring):
                param = findname(param, self.nsview)
            if isinstance(param, ExpressionsEntry):
                param = param.get()
            if param is not None:
                if cfg.debug_bindings:
                    print "binding", v.name(), 'of', type(obj).__name__, 'to', type(param).__name__, '.'.join([param.ns.path, param.name()])
                v.bind(param.getVariable())
            else:
                msg = "unable to bind variable %s of %r" % (v.name(), obj)
                if not v.required():
                    msg += ", optional"
                    print msg
                else:
                    raise Exception(msg)
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
        if ':' in objspec:
            objtype, objpath = objspec.split(":", 1)
            return self.gettype(objtype)[objpath]
        elif '/' in objspec:
            nspath, obsname = objspec.rsplit("/", 1)
            return self.ns(nspath).observables[obsname]
        else:
            return findname(objspec, self.globalns)

env = _environment()
