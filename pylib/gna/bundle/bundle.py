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

def init_bundle(**kwargs):
    names = kwargs.pop('name', None)
    if not names:
        cfg = kwargs['cfg']
        names = cfg.bundle

    if not isinstance( names, (list, tuple) ):
        names = names,

    bundles = ()
    for name in names:
        if ':' in name:
            name, args = name.split(':', 1)
            args = args.split(':')
        else:
            args=[]
        bundleclass = get_bundle(name)
        bundle = bundleclass(*args, **kwargs)
        bundles+=bundle,

    if not bundles:
        raise Exception( "Bundle '%s' is not defined"%str(name) )

    return bundles

def execute_bundle(**kwargs):
    bundles = init_bundle(**kwargs )
    for bundle in bundles:
        bundle.execute()
    return bundles

class TransformationBundle(object):
    """TransformationBundle is a base class for implementing a Bundle.

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
    def __init__(self, cfg, **kwargs):
        """Constructor.

        Arguments:
            - cfg — bundle configuration (NesteDict).

        Keyword arguments:
            - common_namespace — namespace, common for all transformations.
            - namespaces — list of namespaces to create replica of a chain. If a list of strings is passed
              it is replaces by a list of namespaces with corresponding names with parent=common_namespace.
        """
        self.cfg = cfg

        self.common_namespace = kwargs.pop( 'common_namespace', env.globalns )
        namespaces=kwargs.pop('namespaces', None) or [self.common_namespace]
        self.namespaces = [ self.common_namespace(ns) if isinstance(ns, basestring) else ns for ns in namespaces ]

        self.objects             = NestedDict() # {'group': {key: object}} - objects with transformations
        self.transformations_in  = NestedDict() # {key: transformation}    - transformations, that require inputs to be connected
        self.transformations_out = NestedDict() # {key: transformation}    - transformations, with open outputs
        self.outputs             = NestedDict() # {key: output}            - inputs to be connected (should be consistent with transformations_out)
        self.inputs              = NestedDict() # {key: input}             - open outputs (should be consistent with transformations_in)

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


