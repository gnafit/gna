class resolver(object):
    def __init__(self, env, nsview):
        self.env = env
        self.nsview = nsview

    def isbound(self, p):
        return not isinstance(p, str)

    def findpath(self, ns, name, resolving, known):
        entry = ns.getentry(name)
        if entry.get('parameter') is not None:
            return []
        if entry.get('alias'):
            if not isinstance(dep, basestring) or '.' in dep:
                return []
            return self.findpath(ns, entry.get('alias'), resolving, known)
        try:
            exprs = entry['expressions']
        except KeyError:
            return None
        missings = []
        for idx, (obj, expr, deps) in enumerate(exprs):
            missing = []
            for dep in deps:
                if dep in known:
                    continue
                if not isinstance(dep, basestring) or '.' in dep:
                    continue
                missing.append(dep)
            if not missing:
                known.add(name)
                return [(name, idx)]
            missings.append(missing)

        res = (None, None)
        for idx, missing in enumerate(missings):
            cknown = set(known)
            cpaths = []
            for dep in missing:
                if dep in resolving:
                    break
                deppath = self.findpath(ns, dep, resolving.union([dep]), cknown)
                if deppath is None:
                    break
                cpaths.extend(deppath)
            else:
                cpaths.append((name, idx))
                rank = -len(cpaths)
                if res[0] is not None:
                    res = min(res, (rank, cpaths), key=itemgetter(0))
                else:
                    res = (rank, cpaths)
        return res[1]

    def findbinding(self, bindings, varname, resolve, ns=None):
        if '.' in varname:
            nspath, name = varname.rsplit('.', 1)
            ns = self.env.ns(nspath.lstrip('.'))
            varname = name
        if ns is None:
            try:
                binding = next((bs[varname] for bs in reversed(bindings) if varname in bs))
            except StopIteration:
                nsview = self.nsview
            else:
                if isinstance(binding, basestring):
                    return self.findbinding(bindings, binding, resolve, ns=ns)
                else:
                    return binding
        else:
            nsview = ns
        entry = nsview.getentry(varname)
        v = entry.get('alias')
        if v is not None:
            return self.findbinding(bindings, v, resolve, ns=ns)
        v = entry.get('parameter')
        if v is not None:
            return v
        v = entry.get('evaluable')
        if v is not None:
            return v
        if not resolve:
            return None
        path = self.findpath(entry.ns, varname, set([varname]), set())
        if path is None:
            return None
        for name, idx in reversed(path):
            obj, expr, deps = entry.ns.getentry(name)['expressions'][idx]
            varnames = []
            bindings = {}
            for src, dep in zip(expr.sources.itervalues(), deps):
                if self.isbound(dep):
                    bindings[src.name] = dep
                varnames.append(src.name)
            self.resolveobject(obj, varnames=varnames, resolve=False,
                               bindings=[bindings])
            entry.ns.addevaluable(name, expr.get())
        return self.nsview.getentry(varname)['evaluable']

    def resolveobject(self, obj, freevars=(), resolve=True,
                      varnames=None, bindings=[]):
        bound = set()
        for v in obj.variables.itervalues():
            if v.name in freevars:
                continue
            if varnames is not None:
                if v.name not in varnames:
                    continue
            if not v.isFree():
                continue
            found = False
            param = self.findbinding(bindings, v.name, resolve)
            if param is not None:
                print "binding", v.name, 'of', type(obj).__name__, 'to', type(param).__name__, '.'.join([param.ns.path, param.name()])
                v.bind(param.getVariable())
                bound.add(v.name)
            else:
                msg = "unable to bind variable %s of %r" % (v.name, obj)
                if not v.required():
                    msg += ", optional"
                    print msg
                else:
                    raise Exception(msg)
        if varnames is not None:
            diff = bound.difference(varnames)
            if diff:
                msg = "unbound vars:", diff
                raise Exception(msg)
        return obj
