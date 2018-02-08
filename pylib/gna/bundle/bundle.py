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

    bundles = tuple(get_bundle(name) for name in names)
    if not bundles:
        raise Exception( "Bundle '%s' is not defined"%str(name) )

    return tuple(bundle(**kwargs) for bundle in bundles)

def execute_bundle(**kwargs):
    bundles = init_bundle(**kwargs )
    for bundle in bundles:
        bundle.execute()
    return bundles

class TransformationBundle(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg

        self.common_namespace = kwargs.pop( 'common_namespace', env.globalns )
        namespaces=kwargs.pop('namespaces', None) or [self.common_namespace]
        self.namespaces = [ self.common_namespace(ns) if isinstance(ns, basestring) else ns for ns in namespaces ]

        self.transformations     = NestedDict() # {'group': {key: transformation}} - transformations, not listed in transformation_in and transformations_out
        self.transformations_in  = NestedDict() # {key: transformation}            - transfromations, that require inputs to be connected
        self.transformations_out = NestedDict() # {key: transformation}            - transformations, with oupen outputs
        self.outputs             = NestedDict() # {key: output}                    - inputs to be connected (should be consistent with transformations_out)
        self.inputs              = NestedDict() # {key: input}                     - open outputs (should be consistent with transformations_in)

    def execute(self):
        self.define_variables()
        self.build()

    def build(self):
        pass

    def define_variables(self):
        pass

