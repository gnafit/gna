from __future__ import print_function
from gna.env import namespace, env
from gna.config import cfg
from pkgutil import iter_modules

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
    name = kwargs.pop('name', None)
    if not name:
        cfg = kwargs['cfg']
        name = cfg.bundle

    bundle = get_bundle(name)
    if not bundle:
        raise Exception( "Bundle '%s' is not defined"%name )

    return bundle(**kwargs)

def execute_bundle(**kwargs):
    bundle = init_bundle(**kwargs )
    bundle.define_variables()
    bundle.build()
    return bundle

class TransformationBundle(object):
    name = '<undefined>'
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg

        namespaces=kwargs.pop( 'namespaces', [env.globalns] )
        self.namespaces = [ env.globalns(ns) if isinstance(ns, basestring) else ns for ns in namespaces ]

        self.common_namespace = kwargs.pop( 'common_namespace', env.globalns )

        storage=kwargs.pop( 'storage', None )
        if storage:
            self.storage = storage( self.name )
        else:
            self.storage = namespace( None, self.name )

        self.output_transformations = ()
        self.outputs = ()
        self.inputs = ()

    def build(self):
        pass

    def define_variables(self):
        pass
