# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.env import namespace, env
from gna.config import cfg
from pkgutil import iter_modules
from gna.configurator import NestedDict

bundle_modules = {}
bundles = {}

def get_bundle(name):
    if name in bundles:
        return bundles[name]

    if not bundle_modules:
        for bundlepath in cfg.bundlepaths:
            for loader, lname, _ in iter_modules([bundlepath]):
                # print( 'init', lname, loader )
                bundle_modules.update({lname: loader})

    loader = bundle_modules.get( name )
    if not loader:
        raise Exception( 'There is no bundle module for %s in %s'%(name, str(cfg.bundlepaths)) )

    module = loader.find_module(name).load_module(name)
    bundle = getattr( module, name, None )
    if not bundle:
        raise Exception( 'There is no bundle %s in it\'s module'%(name) )
    bundles[name] = bundle

    return bundle

def init_bundle(cfg, *args, **kwargs):
    names = kwargs.pop('name', None)
    if not names:
        names = cfg.bundle

        if isinstance(names, (dict, NestedDict)):
            names, version=(names['name'],), names.get('version', None)
            if version:
                names+=str(version),
            names = '_'.join(names),

    if not isinstance(names, (list, tuple)):
        names = names,

    bundles = ()
    for name in names:
        bundleclass = get_bundle(name)
        bundle = bundleclass(cfg, *args, **kwargs)
        bundles+=bundle,

    if not bundles:
        raise Exception( "Bundle '%s' is not defined"%str(name) )

    return bundles

def execute_bundles(cfg, *args, **kwargs):
    bundles = init_bundle(cfg, **kwargs)
    for bundle in bundles:
        bundle.execute()
    return bundles

def execute_bundle(cfg, *args, **kwargs):
    bundle = init_bundle(cfg, *args, **kwargs)[0]
    bundle.execute()
    return bundle

class TransformationBundleLegacy(object):
    """TransformationBundleLegacy is a base class for implementing a Bundle.

    Bundle is an object that is able to configure and construct a part of a computational chain.
    Bundles should be designed to be backwards compatible. Once bundle is added to the master, its
    code should not be modified and new bundles should be created to introduce additional functionality.

    Bundles are meant to operate on a list of namespaces creating (partial) replicas of transformation chains.
    Distinct namespace may represent one of the similar designed detectors for example.

    The bundle is defined by implementation of two methods:
      - define_variables() — for initializing relevant parameters and variables.
      - build() — for building an actual chain.

    The bundle provides the following data:
        self.objects             - {'group': {key: object}} - objects with transformations
        self.transformations_in  - {key: transformation}    - transformations, that require inputs to be connected
        self.transformations_out - {key: transformation}    - transformations, with open outputs
        self.outputs             - {key: output}            - inputs to be connected (should be consistent with transformations_out)
        self.inputs              - {key: input}             - open outputs (should be consistent with transformations_in)
    """
    def __init__(self, cfg, *args, **kwargs):
        """Constructor.

        Arguments:
            - cfg — bundle configuration (NestedDict).

        Keyword arguments:
            - common_namespace — namespace, common for all transformations.
            - namespaces — list of namespaces to create replica of a chain. If a list of strings is passed
              it is replaces by a list of namespaces with corresponding names with parent=common_namespace.
        """
        self.shared=kwargs.pop('shared', NestedDict())
        self.cfg = cfg

        self.common_namespace = kwargs.pop( 'common_namespace', self.shared.get('common_namespace', env.globalns) )
        namespaces=kwargs.pop('namespaces', self.shared.get('namespaces', None)) or [self.common_namespace]
        self.namespaces = [ self.common_namespace(ns) if isinstance(ns, basestring) else ns for ns in namespaces ]

        self.shared.setdefault('namespaces', self.namespaces)
        self.shared.setdefault('common_namespace', self.common_namespace)

        self.objects             = NestedDict() # {'group': {key: object}} - objects with transformations
        self.transformations_in  = NestedDict() # {key: transformation}    - transformations, that require inputs to be connected
        self.transformations_out = NestedDict() # {key: transformation}    - transformations, with open outputs
        self.outputs             = NestedDict() # {key: output}            - inputs to be connected (should be consistent with transformations_out)
        self.inputs              = NestedDict() # {key: input}             - open outputs (should be consistent with transformations_in)

        self.context = kwargs.pop('context', None)
        if self.context is not None:
            self.set_output = self.context.set_output
            self.set_input  = self.context.set_input
        else:
            self.set_output = lambda *a, **kw: None
            self.set_input  = lambda *a, **kw: None

    def execute(self):
        """Calls sequentially the methods to define variables and build the computational chain."""
        try:
            self.define_variables()
        except Exception as e:
            print( 'Failed to define variables for bundle %s'%(type(self).__name__) )
            import sys
            raise e, None, sys.exc_info()[2]

        try:
            self.build()
        except Exception as e:
            print( 'Failed to build the bundle %s'%(type(self).__name__) )
            import sys
            raise e, None, sys.exc_info()[2]

    def build(self):
        """Builds the computational chain. Should handle each namespace in namespaces."""
        pass

    def define_variables(self):
        """Defines the variables necessary for the computational chain. Should handle each namespace."""
        pass

    def addcfgobservable(self, ns, obj, defname=None, key='observable', fmtdict={}, **kwargs):
        obsname = None

        if key:
            obsname = self.cfg.get(key, None)

        if obsname and isinstance(obsname, bool):
            obsname=defname

        if not obsname:
            return

        if fmtdict:
            obsname=obsname.format(**fmtdict)

        ns.addobservable(obsname, obj, **kwargs)

    def init_indices(self, indices='indices'):
        if isinstance(indices, str):
            self.idx = self.cfg.get(indices, [])
        else:
            self.idx = indices

        from gna.expression import NIndex
        if isinstance(self.idx, NIndex):
            return

        self.idx = NIndex(fromlist=self.idx)

    def exception(self, message):
        return Exception("{bundle}: {message}".format(bundle=type(self).__name__, message=message))

class TransformationBundle(object):
    """
    cfg = {
        bundle = name or {
            name    = string                                             # Bundle name (may contain version)
            version = string (optional)                                  # Bundle version (will be added to bundle name)
            names   = { localname: globalname, localname1: globalname1 } # Map to change the predefined names
            nidx    = NIndex or NIndex configuration list (optional)     # Multiindex, if not passed, empty index is created
            major   = [ 'short1', 'short2', ... ]                        # A subset of indices considered to be major
        }
    }
    """
    def __init__(self, cfg, *args, **kwargs):
        from gna.expression import NIndex
        self.cfg = cfg

        # Read bundle configuration
        self.bundlecfg = cfg.get('bundle', '')

        # If only name is provided, make a dictionary
        if isinstance(self.bundlecfg, str):
            self.bundlecfg=dict(name=self.bundlecfg)

        # Create dictionary to map names if needed
        self.bundlecfg.setdefault('names', {})

        # Init multidimensional index
        self.nidx=self.bundlecfg.get('nidx', [])
        if isinstance(self.nidx, (tuple,list)):
            self.nidx=NIndex(fromlist=self.nidx)
        assert isinstance(self.nidx, NIndex)

        # If information about major indexes is provided, split nidx into major and minor parts
        major = self.bundlecfg.get('major', None)
        if major is not None:
            self.nidx_major, self.nidx_minor = self.nidx.split( major )
        else:
            self.nidx_major, self.nidx_minor = self.nidx, self.nidx.get_subset(())

        # Init namespace and context
        self.context   = kwargs.pop('context', None)
        self.namespace = kwargs.pop('namespace', env.globalns)
        if self.context is None:
            self.context = NestedDict(inputs={}, outputs={}, objects={})

        assert not kwargs

    def execute(self):
        """Calls sequentially the methods to define variables and build the computational chain."""
        try:
            self.define_variables()
        except Exception as e:
            print( 'Failed to define variables for bundle %s'%(type(self).__name__) )
            import sys
            raise e, None, sys.exc_info()[2]

        try:
            self.build()
        except Exception as e:
            print( 'Failed to build the bundle %s'%(type(self).__name__) )
            import sys
            raise e, None, sys.exc_info()[2]

    def build(self):
        """Builds the computational chain. Should handle each namespace in namespaces."""
        pass

    def define_variables(self):
        """Defines the variables necessary for the computational chain. Should handle each namespace."""
        pass

    def exception(self, message):
        return Exception("{bundle}: {message}".format(bundle=type(self).__name__, message=message))

    def get_path(self, localname, nidx=None, argument_number=None, join=False, extra=None):
        name=self.bundlecfg['names'].get(localname, localname)

        if nidx is None:
            path = name,
        else:
            path = nidx.current_values(name=name)

        if extra:
            if isinstance(extra, str):
                path+=extra,
            elif isinstance(extra, (list,tuple)):
                path+=tuple(extra)
            else:
                raise Exception('Unsupported extra field: '+str(extra))

        if argument_number is not None:
            path+=('{:02d}'.format(int(clone)),)

        if join:
            path = '.'.join(path)

        return path

    def reqparameter(self, name, nidx, *args, **kwargs):
        return self.namespace.reqparameter(self.get_path(name, nidx, extra=kwargs.pop('extra', None)), *args, **kwargs)

    def set_output(self, name, nidx, output, extra=None):
        self.outputs[self.get_path(name, nidx, extra=extra)]=output

    def set_input(self, name, nidx, input, argument_number=None, extra=None):
        self.inputs[self.get_path(name, nidx, argument_number, extra=extra)]=input

    def check_nidx_dim(self, dmin, dmax=float('inf'), nidx='both'):
        if nidx=='both':
            nidx=self.nidx
        elif nidx=='major':
            nidx=self.nidx_major
        elif nidx=='minor':
            nidx=self.nidx_minor
        else:
            raise self.exception('Unknown nidx type '+nidx)

        ndim = nidx.ndim();
        if not dmin<=ndim<=dmax:
            raise self.exception('Ndim %i does not satisfy requirement: {}<=ndim<={}'.format(ndim, dmin, dmax))

