from gna.env import env

def hygienic(decorator):
    def new_decorator(original):
        wrapped = decorator(original)
        wrapped.__name__ = original.__name__
        wrapped.__doc__ = original.__doc__
        wrapped.__module__ = original.__module__
        return wrapped
    return new_decorator

@hygienic
def wrapGNAclass(cls):
    class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
            super(WrappedClass, self).__init__(*args)
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
        def __getattr__(self, attr):
            try:
                return self[attr]
            except KeyError:
                raise AttributeError(attr)
    return WrappedClass

def patchSimpleDict(cls):
    def itervalues(self):
        for i in range(self.size()):
            yield self.at(i)

    def values(self):
        return list(itervalues(self))

    def iterkeys(self):
        for i in range(self.size()):
            yield self.at(i).name

    def __contains__(self, key):
        return key in keys(self)

    def keys(self):
        return list(iterkeys(self))

    def iteritems(self):
        for i in range(self.size()):
            yield self.at(i).name, self.at(i)

    def __len__(self):
        return self.size()

    def __iter__(self):
        return self.iterkeys()

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    cls.itervalues = itervalues
    cls.values = values
    cls.iterkeys = iterkeys
    cls.keys = keys
    cls.iteritems = iteritems
    cls.__contains__ = __contains__
    cls.__len__ = __len__
    cls.__iter__ = __iter__
    cls.__getattr__ = __getattr__
    cls.get = get

def patchVariableDescriptor(cls):
    origclaim = cls.claim
    def wrappedclaim(self):
        return ROOT.ParameterWrapper(origclaim(self))
    cls.claim = wrappedclaim

def patchTransformationDescriptor(cls):
    def __getattr__(self, attr):
        inp = getattr(self.inputs, attr, None)
        out = getattr(self.outputs, attr, None)
        if inp and out:
            raise Exception("{} is both input and output".format(attr))
        if inp:
            return inp
        if out:
            return out
        raise AttributeError(attr)

    def __getitem__(self, attr):
        inp = self.inputs.get(attr)
        out = self.inputs.get(attr)
        if inp and out:
            raise Exception("{} is both input and output".format(attr))
        if inp:
            return inp
        if out:
            return out
        raise KeyError(attr)

    cls.__getattr__ = __getattr__
    cls.__getitem__ = __getattr__

def setup(ROOT):
    ROOT.UserExceptions.update({
        "KeyError": KeyError,
        "IndexError": IndexError,
    })

    simpledicts = [
        ROOT.GNAObject.Variables,
        ROOT.GNAObject.Evaluables,
        ROOT.GNAObject.Transformations,
        ROOT.TransformationDescriptor.Inputs,
        ROOT.TransformationDescriptor.Outputs,
        ROOT.EvaluableDescriptor.Sources,
    ]
    for cls in simpledicts:
        patchSimpleDict(cls)

    patchVariableDescriptor(ROOT.VariableDescriptor)
    patchTransformationDescriptor(ROOT.TransformationDescriptor)

    def patchcls(cls):
        if not isinstance(cls, ROOT.PyRootType):
            return cls
        if 'Class' not in cls.__dict__:
            t = cls.__class__
            f = lambda s, n, old=t.__getattribute__: patchcls(s, old(s, n))
            t.__getattribute__ = f
            return cls
        try:
            if cls.Class().InheritsFrom("GNAObject"):
                return wrapGNAclass(cls)
        except AttributeError:
            pass
        return cls

    t = type(ROOT)
    origgetattr = t.__getattr__
    def patchclass(self, name):
        cls = patchcls(origgetattr(self, name))
        self.__dict__[name] = cls
        return cls
    t.__getattr__ = patchclass
