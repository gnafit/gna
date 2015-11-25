class resolver(object):
    def __init__(self, env, nsview):
        self.env = env
        self.nsview = nsview

    def isbound(self, p):
        return not isinstance(p, str)

    def findpath(self, name, seen, known):
        try:
            cands = self.nsview.getexpressions(name)
        except KeyError:
            return None, None
        missings = []
        for i, (obj, expr, deps, ns) in enumerate(cands):
            missing = [x for x in deps
                       if x not in known and not self.isbound(x)]
            if not missing:
                return [(name, i)], set([name])
            missings.append((missing, known))
        minrank = None
        for i, (missing, lknown) in enumerate(missings):
            cknown = set()
            cpaths = [(name, i)]
            for x in missing:
                if x in seen:
                    break
                deppath, newknown = self.getpath(x, seen+[x], lknown)
                if deppath is None:
                    break
                cknown.update(newknown)
                cpaths.extend(deppath)
            else:
                rank = len(cpaths)
                if minrank is None or minrank > rank:
                    minrank = rank
                    minknown = cknown
                    minpaths = cpaths
        if minrank is not None:
            return minpaths, minknown
        else:
            return None, None

    def getpath(self, name, seen, known):
        path, newknown = self.findpath(name, seen, known)
        return path

    def findbinding(self, bindings, varname, resolve, ns=None):
        if '.' in varname:
            nspath, name = varname.rsplit('.', 1)
            ns = self.env.ns(nspath)
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
        try:
            return self.findbinding(bindings, nsview.getalias(varname), resolve, ns=ns)
        except KeyError:
            pass
        try:
            return nsview.getparameter(varname)
        except KeyError:
            pass
        try:
            return nsview.getevaluable(varname)
        except KeyError:
            pass
        if not resolve:
            return None
        path = self.getpath(varname, [varname], nsview.names())
        if path is None:
            return None
        for name, idx in reversed(path):
            obj, expr, deps, ns = self.nsview.getexpressions(name)[idx]
            varnames = []
            bindings = {}
            for src, dep in zip(expr.sources.itervalues(), deps):
                if self.isbound(dep):
                    bindings[src.name] = dep
                varnames.append(src.name)
            self.resolveobject(obj, varnames=varnames, resolve=False,
                               bindings=[bindings])
            self.env.addevaluable(ns, name, expr.get())
        return self.nsview.getevaluable(varname)

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
                print "binding", v.name, 'of', type(obj).__name__, 'to', type(param).__name__, '.'.join([param.ns.path(), param.name()])
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
