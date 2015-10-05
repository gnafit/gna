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

class parameters(object):
    def __init__(self):
        self._exprobjects = []
        self.evaluables = {}
        self.resolver = parresolver(self)
        self.parameters = {}

    def define(self, name, **kwargs):
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
        assert name not in self.parameters
        self.parameters[name] = param
        self.resolver.addparameter(param)
        return param

    def _evaluable(self, name, var):
        evaluable = ROOT.GaussianValue("double")(name, var)
        self.evaluables[name] = evaluable
        self.resolver.addevaluable(evaluable)

    def addexpressions(self, obj, bindings={}):
        self._exprobjects.append(obj)
        for expr in obj.evaluables:
            self.resolver.addexpression(obj, expr, bindings=bindings)

    def resolve(self, obj, **kwargs):
        return self.resolver.resolve(obj, **kwargs)

    def __getitem__(self, name):
        return self.parameters[name]

class parresolver(object):
    def __init__(self, pars):
        self._names = {}
        self._expressions = {}
        self._bindings = {}
        self.pars = pars

    def _isparameter(self, p):
        return not isinstance(p, str)

    def addobject(self, obj):
        self._objects.append(obj)

    def addparameter(self, param):
        self._names[param.name()] = param

    def addevaluable(self, evaluable):
        self._names[evaluable.name()] = evaluable

    def addexpression(self, obj, expr, bindings={}):
        if expr.name not in self._expressions:
            self._expressions[expr.name] = []
        deps = [bindings.get(name, name) for name in expr.sources.iternames()]
        self._expressions[expr.name].append((obj, expr, deps))

    def getpath(self, name, seen, known):
        try:
            cands = self._expressions[name]
        except KeyError:
            return None, None
        missings = []
        for i, (obj, expr, deps) in enumerate(cands):
            missing = [x for x in deps
                       if x not in known and not self._isparameter(x)]
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

    def findbinding(self, obj, varname, resolve):
        try:
            return self._names[varname]
        except KeyError:
            pass
        if not resolve:
            return None
        path, _ = self.getpath(varname, [varname], set(self._names.keys()))
        if path is None:
            return None
        for name, idx in reversed(path):
            obj, expr, deps = self._expressions[name][idx]
            varnames = []
            bindings = {}
            for src, dep in zip(expr.sources, deps):
                if self._isparameter(dep):
                    bindings[src.name] = dep
                varnames.append(src.name)
            self.resolveobject(obj, varnames=varnames, resolve=False,
                               bindings=bindings)
            self.pars._evaluable(name, expr.get())
        return self._names[varname]

    def resolveobject(self, obj, freevars=(), resolve=True,
                      varnames=None, bindings={}):
        bound = set()
        for v in obj.variables:
            if v.name in freevars:
                continue
            if varnames is not None:
                if v.name not in varnames:
                    continue
            if not v.isFree():
                continue
            found = False
            binding = bindings.get(v.name, v.name)
            if not self._isparameter(binding):
                param = self.findbinding(obj, binding, resolve=resolve)
            else:
                param = binding
            if param is not None:
                v.bind(param.getVariable())
                bound.add(v.name)
            else:
                msg = "unbound variable %s on %r" % (v.name, obj)
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

    def resolve(self, obj, **kwargs):
        if isinstance(obj, ROOT.ExpressionsProvider):
            return
        self.resolveobject(obj, **kwargs)
