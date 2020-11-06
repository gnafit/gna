import numpy as np
import ROOT
import itertools as it
import types
import inspect

# breaking change in ROOT 6.22 >= due to new PyROOT
try:
    import cppyy
    Template = cppyy._cpython_cppyy.Template
    def istemplate(cls):
        return isinstance(cls, Template)
except AttributeError:
    # ROOT <= 6.22 or 6.22 with legacy PyROOT
    def istemplate(cls):
        return not isinstance(cls, ROOT.PyRootType)


ROOT.GNAObjectT
provided_precisions = [str(prec) for prec in ROOT.GNA.provided_precisions()]

def patchGNAclass(cls):
    if '__original_init__' in cls.__dict__:
        return cls
    def newinit(self, *args, **kwargs):
        self.__original_init__(*args)
        if not self:
            return
        labels = kwargs.pop('labels', None) or []
        bind = kwargs.pop('bind', True)
        freevars = kwargs.pop('freevars', ())
        bindings = kwargs.pop('bindings', {})
        ns = kwargs.pop('ns', None)
        if kwargs:
            msg = "unknown keywords %s in constructor of %r"
            msg = msg % (', '.join(kwargs.keys()), self)
            raise Exception(msg)
        from gna.env import env
        env.register(self, bind=bind, freevars=freevars,
                     ns=ns, bindings=bindings)

        if isinstance(labels, str):
            labels = [labels]
        for trans, label in zip(self.transformations.values(), labels):
            trans.setLabel(label)

    cls.__original_init__, cls.__init__ = cls.__init__, newinit

    return cls

class GNAObjectTemplates(object):
    """Patch GNA object templates"""
    def __init__(self, parent, name):
        self.namespace=getattr(parent, name)
        setattr(parent, name, self)

    def __getattr__(self, name):
        ret = getattr(self.namespace, name)
        if name.startswith('__'):
            return ret
        self.patchGNATemplate(ret)
        setattr(self, name, ret)
        return ret

    @staticmethod
    def patchGNATemplate(template):
        if inspect.isclass(template):
            return

        for pp in provided_precisions:
            cls = template(pp)
            patchGNAclass(cls)

GNAObjectTemplates=GNAObjectTemplates(ROOT.GNA, 'GNAObjectTemplates')

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
        return iter(self.keys())

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

    def __dir__(self):
        return dir(cls) + list(self.keys())

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
    cls.__dir__ = __dir__

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

def patchDescriptor(cls):
    def __hash__(self):
        return self.hash()
    def __eq__(self, other):
        return self.hash() == other.hash()
    cls.__hash__ = __hash__
    cls.__eq__ = __eq__

def importcommon():
    from gna.bindings import DataType, OutputDescriptor, InputDescriptor, TransformationDescriptor, GNAObject, Variable

def legacytypes():
    ROOT.TransformationTypes.OutputHandle=ROOT.TransformationTypes.OutputHandleT('double')
    ROOT.TransformationTypes.InputHandle=ROOT.TransformationTypes.InputHandleT('double')

def setup(ROOT):
    if hasattr( ROOT, '__gna_patched__' ) and ROOT.__gna_patched__:
        return
    ROOT.__gna_patched__ = True
    ROOT.GNAObjectT

    simpledicts=[]
    for ft in provided_precisions:
        obj = ROOT.GNAObjectT(ft, ft)
        descr = ROOT.TransformationDescriptorT(ft, ft)
        simpledicts += [
            obj.Variables,
            obj.Evaluables,
            obj.Transformations,
            descr.Inputs,
            descr.Outputs,
        ]
    simpledicts+=[ROOT.EvaluableDescriptor.Sources]
    for cls in simpledicts:
        patchSimpleDict(cls)

    patchVariableDescriptor(ROOT.VariableDescriptor)
    for ft in provided_precisions:
        patchTransformationDescriptor(ROOT.TransformationDescriptorT(ft, ft))
        patchDescriptor(ROOT.InputDescriptorT(ft, ft))
        patchDescriptor(ROOT.OutputDescriptorT(ft, ft))

    patchStatistic(ROOT.Statistic)

    # Protect the following classes/namespaces from being wrapped
    ignored_classes = [
            'Eigen',
            'EigenHelpers',
            'DataType',
            'GNA',
            'GNAUnitTest',
            'NeutrinoUnits',
            'TransformationTypes',
            'ParametrizedTypes',
            'TypesFunctions',
            'TypeClasses',
            'TMath',
            ]

    GNAObjectBase = ROOT.GNAObjectT('void', 'void')
    def patchcls(cls):
        if not inspect.isclass(cls):
            return cls
        if istemplate(cls):
            return cls
        if cls.__name__.endswith('_meta') or cls.__name__ in ignored_classes:
            return cls
        if issubclass(cls, GNAObjectBase):
            wrapped = patchGNAclass(cls)
            return wrapped
        return cls

    t = type(ROOT)
    origgetattr = t.__getattr__
    def patchclass(self, name):
        try:
            # modern PyROOT
            cls = patchcls(origgetattr(name))
        except TypeError:
            # legacy PyROOT
            cls = patchcls(origgetattr(self, name))

        self.__dict__[name] = cls
        return cls
    t.__getattr__ = patchclass

    importcommon()
    legacytypes()

def patchROOTClass(classes=None, methods=None):
    """Decorator to override ROOT class methods. Usage
    @patchclass
    def CLASSNAME__METHODNAME(self,...)

    @patchclass(CLASSNAME, METHODNAME)
    def function(self,...)

    @patchclass(ROOT.CLASS, METHODNAME)
    def function(self,...)

    @patchclass( [ROOT.CLASS1, ROOT.CLASS2,...], METHODNAME )
    def function(self,...)
    """
    function=None

    # Used as decorator
    if isinstance(classes, types.FunctionType):
        function = classes
        classname, method = function.__name__.split( '__', 1 )
        setattr(getattr(ROOT, classname), method, function)
        return classes

    # Used as function returning decorator
    if not isinstance(classes, (list, tuple)):
        classes = (classes,)

    classes = [getattr(ROOT, o) if isinstance(o, str) else o for o in classes]

    if methods is not None and not isinstance(methods, (list, tuple)):
        methods = (methods,)

    def decorator(function):
        lmethods = methods
        if lmethods is None:
            lmethods = function.__name__.split( '__', 1 )[1],
        for cls in classes:
            for method in lmethods:
                origname = '__%s_orig'%method
                origmethod = getattr(cls, method, None)
                if origmethod:
                    setattr(cls, origname, origmethod)
                setattr(cls, method, function)
        return function

    return decorator

def patchROOTTemplate(templates=None, methods=None):
    if not isinstance(templates, tuple):
        templates = (templates,)
    classes = tuple(template(precision) for template in templates for precision in provided_precisions)
    return patchROOTClass(classes, methods)
