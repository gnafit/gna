from __future__ import print_function
from collections import defaultdict, deque, Mapping, OrderedDict
import parameters
from contextlib import contextmanager
import ROOT
from gna.config import cfg

provided_precisions = list(ROOT.GNA.provided_precisions())
expressionproviders = tuple(ROOT.GNA.GNAObjectTemplates.ExpressionsProviderT(p) for p in provided_precisions)

env = None

class namespacedict(OrderedDict):
    def __init__(self, ns):
        super(namespacedict, self).__init__()
        self.ns = ns

    def __missing__(self, key):
        if key in self.ns.storage:
            return self.ns
        value = namespace(self.ns, key)
        self[key] = value
        return value

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
                    dep = env.nsview[dep]
                except KeyError:
                    return None
            if isinstance(dep, ExpressionsEntry):
                if dep in seen:
                    return None
                path = dep.resolvepath(seen | {dep}, known)
                if path is None:
                    return None
                known.add(dep)
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
                dep = env.nsview[dep]
            v.bind(dep.getVariable())
        return self.ns.addevaluable(self.expr.name(), self.expr.get())

class ExpressionsEntry(object):
    _label = None
    def __init__(self, ns):
        self.ns = ns
        self.exprs = []

    def setLabel(self, label):
        self._label = label

    def label(self):
        return self._label

    def add(self, obj, expr, bindings):
        self.exprs.append(ExpressionWithBindings(self.ns, obj, expr, bindings))

    def get(self):
        path = self.resolvepath({self}, OrderedDict())
        if not path:
            names = [expr.expr.name() for expr in self.exprs]
            reqs = [var.name() for expr in self.exprs for var in expr.expr.sources.values()]
            raise KeyError('Unable to provide required variables for {!s}. Something is missing from: {!s}'.format(names, reqs))
        for expr in path:
            v = expr.get()
            if self._label is not None:
                otherlabel = v.label()
                if otherlabel:
                    newlabel = '{}: {}'.format(self._label, otherlabel)
                else:
                    newlabel = self._label
                v.setLabel(newlabel)
        return v

    materialize = get

    def resolvepath(self, seen, known):
        minexpr, minpaths = None, None
        for expr in self.exprs:
            if cfg.debug_bindings:
                print(expr.expr.name(), seen)
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
    groups = None
    def __init__(self, parent, name):
        self.groups=[]
        self.name = name
        if parent and name=='':
            raise Exception( 'Only root namespace may have no name' )
        if parent:
            self.path = parent.pathto(name)
        else:
            self.path = name

        self.storage = OrderedDict()
        self.observables = OrderedDict()
        self.observables_tags = defaultdict(set)

        self.rules = []
        self.namespaces = namespacedict(self)

        self.objs = []

    def pathto(self, name):
        return '.'.join((self.path, name)) if self.path else name

    def __nonzero__(self):
        return True

    def __repr__(self):
        return "<namespace path='{0}'>".format(self.path)

    def __enter__(self):
        env.nsview.add([self])

    def __exit__(self, type, value, tb):
        env.nsview.remove([self])

    def __call__(self, nsname):
        if not nsname:
            return self

        if isinstance(nsname, basestring):
            if nsname=='':
                return self
            parts = nsname.split('.')
        elif isinstance(nsname, (list,tuple)):
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

    def get_proper_ns(self, name, separator='.'):
        if isinstance(name, (tuple, list)):
            path, head = name[:-1], name[-1]
        else:
            path, head = (), name

        if separator in head:
            newpath = tuple(head.split(separator))
            path+=newpath[:-1]
            head = newpath[-1]

        if path:
            return self(path), head
        else:
            return None, head

    def __getitem__(self, name):
        if not name:
            return self

        ns, head = self.get_proper_ns(name)

        if ns:
            return ns.__getitem__(head)

        v = self.storage[head]
        if isinstance(v, basestring):
            return env.nsview[v]
        return v

    def get(self, name, *args):
        if not name:
            return self

        ns, head = self.get_proper_ns(name)

        if ns:
            return ns.__getitem__(head)

        v = self.storage.get(head, *args)
        if isinstance(v, basestring):
            return env.nsview[v]
        return v

    def __setitem__(self, name, value):
        ns, head = self.get_proper_ns(name)
        if ns:
            ns.__setitem__(head, value)

        self.storage[head] = value

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

    def defparameter(self, name, *args, **kwargs):
        ns, head = self.get_proper_ns(name)
        if ns:
            return ns.defparameter(head, *args, **kwargs)

        if head in self.storage:
            raise Exception("{} is already defined in {}".format(head, self.path))
        target = self.matchrule(head)
        if not target:
            target = kwargs.pop('target', None)
        if target:
            p = target
        else:
            p = parameters.makeparameter(self, head, *args, **kwargs)
        self[head] = p
        return p

    def reqparameter(self, name, *args, **kwargs):
        ns, head = self.get_proper_ns(name)
        if ns:
            return ns.reqparameter(head, *args, **kwargs)

        par = None
        try:
            par = self[head]
        except KeyError:
            pass

        if not par:
            try:
                par = env.nsview[head]
            except KeyError:
                pass

        found=bool(par)
        if not par:
            par = self.defparameter(head, *args, **kwargs)

        if kwargs.get('with_status'):
            return par, found

        return par

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

    def addobservable(self, name, output, export=True, ignorecheck=False):
        ns, head = self.get_proper_ns(name, separator='/')
        if ns:
            return ns.addobservable(head, output, export, ignorecheck)

        if ignorecheck or output.check():
            self.observables[head] = output
            print('Add observable:', '%s/%s'%(self.path, head))
        else:
            print("observation", name, "is invalid")
            output.dump()
        if not export:
            self.observables_tags[name].add('internal')

    def getobservable(self, name):
        ns, head = self.get_proper_ns(name, separator='/')
        if ns:
            return ns.getobservable(head)

        try:
            return self.observables[head]
        except:
            print('Invalid observable', head)

    def addexpressions(self, obj, bindings=[]):
        for expr in obj.evaluables.itervalues():
            if cfg.debug_bindings:
                print(self.path, obj, expr.name())
            name = expr.name()
            if name not in self.storage:
                self.storage[name] = ExpressionsEntry(self)
            if isinstance(self.storage[name], ExpressionsEntry):
                self.storage[name].add(obj, expr, bindings)

    def addevaluable(self, name, var):
        evaluable = ROOT.Variable(var.typeName())(name, var)
        evaluable.setLabel(var.label())
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

    def printobservables(self, internal=False):
        import gna.bindings.DataType
        for path, out in self.walkobservables(internal):
            print('%-30s'%(path+':'), str(out.datatype()))

    def printparameters(self, **kwargs):
        from gna.parameters.printer import print_parameters
        print_parameters(self, **kwargs)

    def materializeexpressions(self, recursive=False):
        for v in self.itervalues():
            if not isinstance(v, ExpressionsEntry):
                continue
            v.materialize()

        if recursive:
            for ns in self.namespaces.values():
                ns.materializeexpressions(True)

    def get_obs(self, *names):
        import fnmatch as fn
        obses = []
        for name in names:
            matched = fn.filter(self.observables.keys(), name)
            obses.extend(matched)
        return obses

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
            print("can't find name {}. Names in view: ".format(name), end='')
            if self.nses:
                for ns in self.nses:
                    print('"{}": "{}"'.format(ns.path, ', '.join(ns.storage)), ' ', end='')
                print('')
            else:
                print('none')
        raise KeyError('%s (namespaces: %s)'%(name, str([ns.name for ns in self.nses])))

    def currentns(self):
        return self.nses[0]

class parametersview(object):
    def __getitem__(self, name):
        res = env.nsview[name]
        return res

    @contextmanager
    def update(self, newvalues={}):
        params=[]
        for p, v in newvalues.iteritems():
            if isinstance(p, str):
                p = self[p]
            p.push(v)
            params.append(p)
        yield
        for p in params:
            p.pop()

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

class PartNotFoundError(Exception):
    def __init__(self, parttype, partname):
        self.parttype = parttype
        self.partname = partname
        msg = "Failed to find {} in the env".format(self.partname)
        super(PartNotFoundError, self).__init__(msg)


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

        from tools.dictwrapper import DictWrapper
        self.future = DictWrapper(OrderedDict(), split='.')

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
        obj.currentns = self.nsview.currentns()
        bindings = self._bindings+[kwargs.pop("bindings", OrderedDict())]
        if ns:
            ns.addexpressions(obj, bindings=bindings)
        if not kwargs.pop('bind', True):
            return obj
        if isinstance(obj, expressionproviders):
            return obj
        freevars = kwargs.pop('freevars', [])

        for v in obj.variables.itervalues():
            if v.name() in freevars:
                continue
            if not v.isFree():
                if cfg.debug_bindings:
                    print('binding skipped', v.name())
                continue
            vname = v.name()
            param = next((bs[vname] for bs in bindings if vname in bs), vname)
            if isinstance(param, basestring):
                param = self.nsview[param]
            if isinstance(param, ExpressionsEntry):
                param = param.get()
            if param is not None:
                if cfg.debug_bindings:
                    print("binding", v.name(), 'of', type(obj).__name__, 'to', type(param).__name__, '.'.join([param.ns.path, param.name()]))
                v.bind(param.getVariable())
            else:
                msg = "unable to bind variable %s of %r" % (v.name(), obj)
                if not v.required():
                    msg += ", optional"
                    print(msg)
                else:
                    raise Exception(msg)
        obj.variablesBound()
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

    # def iternstree(self):
        # return self.globalns.iternstree()

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
            return self.globalns[objspec]

env = _environment()
