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
    return bundle.build(), bundle

class TransformationBundle(object):
    def __init__(self, cfg, namespaces=[env.globalns], storage=None, **kwargs):
        self.cfg = cfg
        self.namespaces = namespaces
        self.storage_name = kwargs.pop( 'storage_name', '' )

        if storage:
            self.storage = storage( self.storage_name )
        else:
            self.storage = namespace( None, storage_name )

    def build(self):
        return None
