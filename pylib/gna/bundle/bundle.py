from __future__ import print_function
from gna.env import namespace, env

bundles_list = dict()

def declare_bundle(name):
    def register_class(cls):
        bundles_list[name] = cls
        return cls
    return register_class

def init_bundle(**kwargs):
    name = kwargs.pop('name', None)
    if not name:
        cfg = kwargs['cfg']
        name = cfg.bundle

    bundle = bundles_list.get(name, None)
    if not bundle:
        print('Available bundles:', sorted(bundles_list.keys()))
        raise Exception( "Bundle '%s' is not defined"%name )

    return bundle(**kwargs)

def execute_bundle(**kwargs):
    bundle = init_bundle(**kwargs )
    bundle.define_variables()
    return bundle.build(), bundle

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

    def build(self):
        pass

    def define_variables(self):
        pass
