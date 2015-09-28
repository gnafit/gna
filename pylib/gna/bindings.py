from gna.env import env

def setup(ROOT):
    simpledicts = [
        ROOT.GNAObject.Variables,
        ROOT.GNAObject.Evaluables,
        ROOT.EvaluableDescriptor.Sources,
    ]

    def itervalues(self):
        for i in range(self.size()):
            yield self.at(i)

    def iternames(self):
        for i in range(self.size()):
            yield self.at(i).name

    for cls in simpledicts:
        cls.__len__ = cls.size
        cls.__iter__ = itervalues
        cls.itervalues = itervalues
        cls.iternames = iternames

    origclaim = ROOT.VariableDescriptor.claim
    def wrappedclaim(self):
        return ROOT.ParameterWrapper(origclaim(self))
    ROOT.VariableDescriptor.claim = wrappedclaim

    def patchcls(cls):
        if not isinstance(cls, ROOT.PyRootType):
            return cls
        if 'Class' not in cls.__dict__:
            t = cls.__class__
            f = lambda s, n, old=t.__getattribute__: patchcls(s, old(s, n))
            t.__getattribute__ = f
            return cls
        try:
            isgnaobj = cls.Class().InheritsFrom("GNAObject")
        except AttributeError:
            return cls
        if not isgnaobj:
            return cls
        class X(cls):
            def __init__(self, *args, **kwargs):
                super(X, self).__init__(*args)
                if not self:
                    return
                bind = kwargs.pop('bind', True)
                freevars = kwargs.pop('freevars', ())
                bindings = kwargs.pop('bindings', {})
                if kwargs:
                    msg = "unknown keywords %s in constructor of %r"
                    msg = msg % (', '.join(kwargs.keys()), self)
                    raise Exception(msg)
                env.current.register(self, bind=bind, freevars=freevars,
                                     bindings=bindings)
        return X

    ROOT.PyRootType
    t = type(ROOT)
    origgetattr = t.__getattr__
    def patchclass(self, name):
        cls = patchcls(origgetattr(self, name))
        self.__dict__[name] = cls
        return cls
    t.__getattr__ = patchclass
