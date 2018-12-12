# -*- coding: utf-8 -*-
from __future__ import print_function

from gna.env import env
import numpy as np
import ROOT

# Protect the following classes/namespaces from being wrapped
ignored_classes = [
        'Eigen',
        'EigenHelpers',
        'DataType',
        'GNA',
        'GNAUnitTest',
        'TransformationTypes',
        'ParametrizedTypes',
        'TypesFunctions',
        ]

def patchGNAclass(cls):
    def newinit(self, *args, **kwargs):
        self.__original_init__(*args)
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
        env.register(self, bind=bind, freevars=freevars,
                     ns=ns, bindings=bindings)

    def newgetattr(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    cls.__original_init__, cls.__init__ = cls.__init__, newinit

    if hasattr(cls, '__getattr__'):
        cls.__original_getattr__, cls.__getattr__ = cls.__getattr__, newgetattr
    else:
        cls.__original_getattr__, cls.__getattr__ = None, newgetattr

    return cls

def patchSimpleDict(cls):
    def itervalues(self):
        for i in range(self.size()):
            yield self.at(i)

    def values(self):
        return list(itervalues(self))

    def iterkeys(self):
        for i in range(self.size()):
            yield self.at(i).name()

    def __contains__(self, key):
        return key in keys(self)

    def keys(self):
        return list(iterkeys(self))

    def iteritems(self):
        for i in range(self.size()):
            yield self.at(i).name(), self.at(i)

    def __len__(self):
        return self.size()

    def __iter__(self):
        return self.iterkeys()

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(attr)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
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
    # origview = cls.view
    def data(self):
        buf = origdata(self)
        datatype = self.datatype()
        return np.frombuffer(buf, count=datatype.size()).reshape(datatype.shape, order='F')
    cls.data = data
    cls.__data_raw__ = origdata
    # cls.__view_raw__ = origview

    # origview = cls.view
    # def view(self):
        # buf = origview(self)
        # datatype = self.datatype()
        # return np.frombuffer(buf, count=datatype.size()).reshape(datatype.shape, order='F')
    # cls.view = view
    # cls.__view_raw__ = origview

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

def patchSingle(single):
    if not hasattr(single, 'single'):
        return
    oldsingle = single.single
    def newsingle(self):
        return ROOT.OutputDescriptor(oldsingle(self))
    single.single = newsingle

def patchStatistic(cls):
    def __call__(self):
        return self.value()
    cls.__call__ = __call__

def patchDescriptor(cls):
    def __hash__(self):
        return self.hash()
    def __eq__(self, other):
        return self.hash() == other.hash()
    cls.__hash__ = __hash__
    cls.__eq__ = __eq__

def importcommon():
    from gna.bindings import DataType, OutputDescriptor, InputDescriptor, TransformationDescriptor, GNAObject

def setup(ROOT):
    if hasattr( ROOT, '__gna_patched__' ) and ROOT.__gna_patched__:
        return
    ROOT.__gna_patched__ = True
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

    patchDescriptor(ROOT.InputDescriptor)
    patchDescriptor(ROOT.OutputDescriptor)

    dataproviders = [
        ROOT.OutputDescriptor,
        ROOT.SingleOutput,
    ]
    for cls in dataproviders:
        patchDataProvider(cls)

    patchSingle( ROOT.GNASingleObject )

    GNAObject = ROOT.GNAObject
    def patchcls(cls):
        if not isinstance(cls, ROOT.PyRootType):
            return cls
        if cls.__name__.endswith('_meta') or cls.__name__ in ignored_classes:
            return cls
        if issubclass(cls, GNAObject):
            wrapped = patchGNAclass(cls)
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

    importcommon()

def patchROOTClass( object=None, method=None ):
    """Decorator to override ROOT class methods. Usage
    @patchclass
    def CLASSNAME__METHODNAME(self,...)

    @patchclass( CLASSNAME, METHODNAME )
    def function(self,...)

    @patchclass( ROOT.CLASS, METHODNAME )
    def function(self,...)

    @patchclass( [ROOT.CLASS1, ROOT.CLASS2,...], METHODNAME )
    def function(self,...)
    """
    cfcn = None
    if not method:
        cfcn = object
        spl = cfcn.__name__.split( '__' )
        object=spl[0]
        method = '__'.join( spl[1:] )

    if not type( object ) is list:
        object = [ object ]

    def converter( fcn ):
        for o in object:
            if type(o)==str:
                o = getattr( ROOT, o )
            setattr( o, method, fcn )
        return fcn

    if cfcn:
        return converter( cfcn )

    return converter
