from gna.env import env
import numpy as np

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
            ns = kwargs.pop('ns', None)
            if kwargs:
                msg = "unknown keywords %s in constructor of %r"
                msg = msg % (', '.join(kwargs.keys()), self)
                raise Exception(msg)
            env.current.register(self, bind=bind, freevars=freevars,
                                 ns=ns, bindings=bindings)
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

def patchDataProvider(cls):
    origdata = cls.data
    origview = cls.view
    def data(self):
        buf = origdata(self)
        datatype = self.datatype()
        return np.frombuffer(buf, count=datatype.size()).reshape(datatype.shape)
    cls.data = data

    origview = cls.view
    def view(self):
        buf = origview(self)
        datatype = self.datatype()
        return np.frombuffer(buf, count=datatype.size()).reshape(datatype.shape)
    cls.view = view

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
        out = self.outputs.get(attr)
        if inp and out:
            raise Exception("{} is both input and output".format(attr))
        if inp:
            return inp
        if out:
            return out
        raise KeyError(attr)

    cls.__getattr__ = __getattr__
    cls.__getitem__ = __getitem__

def patchStatistic(cls):
    def __call__(self):
        return self.value()
    cls.__call__ = __call__

@hygienic
def wrapPoints(cls):
    class WrappedClass(cls):
        def __init__(self, *args, **kwargs):
            if len(args) == 1:
                arr = args[0]
                if isinstance(arr, np.ndarray) or isinstance(arr, list):
                    arr = np.ascontiguousarray(args[0], dtype=np.float64)
                    return super(WrappedClass, self).__init__(arr, len(arr), **kwargs)
                super(WrappedClass, self).__init__(*args, **kwargs)
    return WrappedClass

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

    patchStatistic(ROOT.Statistic)

    dataproviders = [
        ROOT.OutputDescriptor,
        ROOT.SingleOutput,
    ]
    for cls in dataproviders:
        patchDataProvider(cls)

    GNAObject = ROOT.GNAObject
    def patchcls(cls):
        if not isinstance(cls, ROOT.PyRootType):
            return cls
        if cls.__name__.endswith('_meta'):
            return cls
        if issubclass(cls, GNAObject):
            wrapped = wrapGNAclass(cls)
            if cls.__name__ == 'Points':
                wrapped = wrapPoints(wrapped)
            return wrapped
        if 'Class' not in cls.__dict__:
            t = cls.__class__
            origgetattr = cls.__getattribute__
            t.__getattribute__ = lambda s, n: patchcls(origgetattr(s, n))
            return cls
        return cls

    t = type(ROOT)
    origgetattr = t.__getattr__
    def patchclass(self, name):
        cls = patchcls(origgetattr(self, name))
        self.__dict__[name] = cls
        return cls
    t.__getattr__ = patchclass
