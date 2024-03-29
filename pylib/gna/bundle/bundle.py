from gna.env import namespace, env
from gna.config import cfg
from pkgutil import iter_modules
from gna.configurator import NestedDict
import re
from gna.packages import iterate_module_paths
from collections.abc import Mapping

bundle_modules = None
bundles = {}

class BundleException(Exception):
    pass

def get_bundle(name):
    global bundle_modules, bundles
    if isinstance(name, tuple):
        name = '_'.join((o for o in name if o))

    if name in bundles:
        return bundles[name]

    if bundle_modules is None:
        bundle_modules=dict([(lname, loader) for loader, lname, _ in iter_modules(cfg.bundlepaths)]) # TODO: deprecate
        bundle_modules.update([(lname, loader) for loader, lname, _ in iter_modules(iterate_module_paths('bundles'))])

    loader = bundle_modules.get( name )
    if not loader:
        raise Exception( 'There is no bundle module for %s in %s'%(name, str(cfg.bundlepaths)) )

    module = loader.find_module(name).load_module(name)
    bundle = getattr( module, name, None )
    if not bundle:
        raise Exception( 'There is no bundle %s in it\'s module'%(name) )
    bundles[name] = bundle

    return bundle

def get_bundle_by_cfg(cfg):
    bcfg = cfg['bundle']
    return get_bundle((bcfg['name'], bcfg.get('version', None)))

preprocess_mask = re.compile(r'^(.+?)(_(v\d\d))?$')
def preprocess_bundle_cfg(cfg):
    """Check the 'bundle' field and expand it if needed"""

    try:
        bundle = cfg.get('bundle')
    except KeyError:
        raise Exception('Bundle configuration should contain field "bundle"')

    if isinstance(bundle, str):
        name, _, version = preprocess_mask.match(bundle).groups()
        if version is None:
            version=''

        bundle = cfg['bundle'] = NestedDict(name=name, version=version)

    bundle.setdefault('names', {})

def init_bundle(cfg, *args, **kwargs):
    # TODO: remove kwargs['name']
    preprocess_bundle_cfg(cfg)

    names = kwargs.pop('name', None)
    if not names:
        name, version=cfg.bundle['name'], cfg.bundle.get('version', '')
        names = (name, version),

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
        self.namespaces = [ self.common_namespace(ns) if isinstance(ns, str) else ns for ns in namespaces ]

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
            self.common_namespace = self.context.namespace()
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
            raise e.with_traceback(sys.exc_info()[2])

        try:
            self.build()
        except Exception as e:
            print( 'Failed to build the bundle %s'%(type(self).__name__) )
            import sys
            raise e.with_traceback(sys.exc_info()[2])

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

    _debug = False
    def __init__(self, cfg, *args, **kwargs):
        from gna.expression import NIndex
        preprocess_bundle_cfg(cfg)
        self.cfg = cfg

        # Read bundle configuration
        self.bundlecfg = cfg['bundle']

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
        context = kwargs.pop('context', None)
        if context is None:
            inputs  = kwargs.pop('inputs', {})
            outputs = kwargs.pop('outputs', {})
            self.namespace = kwargs.pop('namespace', env.globalns)
        else:
            inputs  = context.inputs
            outputs = context.outputs
            self.namespace = context.namespace
        self.context = NestedDict(inputs=inputs, outputs=outputs, objects={}, namespace=self.namespace)

        self._debug = self.bundlecfg.get('debug', self._debug)

        self._namefunction = self._get_namefunction(self.bundlecfg)

        assert not kwargs, 'Unparsed kwargs: '+str(kwargs)

    @classmethod
    def _get_namefunction(cls, bundlecfg):
        names = bundlecfg.get('names')
        if not names:
             return lambda s: s
        elif isinstance(names, (Mapping, NestedDict)):
            return lambda s: names.get(s, s)
        elif callable(names):
            return names
        elif isinstance(names, (list,tuple)):
            try:
                prefix, postfix = names
            except ValueError:
                pass
            else:
                return lambda s: prefix+s+postfix
        elif isinstance(names, str):
            return lambda s: s+names

        cls.exception('option "names" should be dictionary, function, pair (list, tuple) or a string, not {}'.format(type(names).__name__))

    def execute(self):
        """Calls sequentially the methods to define variables and build the computational chain."""
        try:
            self.define_variables()
        except Exception as e:
            print( 'Failed to define variables for bundle %s'%(type(self).__name__) )
            import sys
            raise e.with_traceback(sys.exc_info()[2])

        try:
            self.build()
        except Exception as e:
            print( 'Failed to build the bundle %s'%(type(self).__name__) )
            import sys
            raise e.with_traceback(sys.exc_info()[2])

    @staticmethod
    def _provides(cfg):
        print('Warning! Calling default bundle.provides(cfg) static method. Should be overridden.')
        print('Bundle configuration:')
        print(str(cfg))
        return (), () # variables, objects

    @classmethod
    def provides(cls, cfg):
        variables, objects = cls._provides(cfg)
        namefcn = cls._get_namefunction(cfg['bundle'])
        ret = tuple((namefcn(a) for a in variables)), tuple((namefcn(a) for a in objects))

        if cfg.bundle.get('debug', False):
            print('Bundle', cls.__name__, 'provides')
            print('  variables', ret[0])
            print('  objects', ret[1])

        return ret

    def build(self):
        """Builds the computational chain. Should handle each namespace in namespaces."""
        pass

    def define_variables(self):
        """Defines the variables necessary for the computational chain. Should handle each namespace."""
        pass

    @classmethod
    def exception(cls, message):
        return Exception("{bundle}: {message}".format(bundle=cls.__name__, message=message))

    def get_globalname(self, localname):
        return self._namefunction(localname)

    def get_path(self, localname, nidx=None, argument_number=None, join=False, extra=None):
        name=self.get_globalname(localname)

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
            path+=('{:02d}'.format(int(argument_number)),)

        if join:
            path = '.'.join(path)

        return path

    def reqparameter(self, name, nidx, *args, **kwargs):
        label = kwargs.get('label', '')
        namespace = kwargs.pop('namespace', self.namespace)
        if nidx is not None and '{' in label:
            kwargs['label'] = nidx.current_format(label)

        path = self.get_path(name, nidx, extra=kwargs.pop('extra', None))
        if self._debug:
            print('{name} requires parameter {par}'.format(name=type(self).__name__, par=path))
        return namespace.reqparameter(path, *args, **kwargs)

    def set_output(self, name, nidx, output, extra=None):
        path=self.get_path(name, nidx, extra=extra)
        if path in self.context.outputs:
            raise Exception('Outputs dictionary already contains '+str(path))

        if self._debug:
            print('{name} sets output {output}'.format(name=type(self).__name__, output=path))
        self.context.outputs[path]=output

    def set_input(self, name, nidx, input, argument_number=None, extra=None):
        path=self.get_path(name, nidx, argument_number, extra=extra)
        if path in self.context.inputs:
            raise Exception('Inputs dictionary already contains '+str(path))

        if self._debug:
            print('{name} sets input {input}'.format(name=type(self).__name__, input=path))
        self.context.inputs[path]=input

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
            raise self.exception('Ndim {} does not satisfy requirement: {}<=ndim<={}'.format(ndim, dmin, dmax))

    def _exception(self, msg):
        return BundleException('{}: {}'.format(type(self).__name__, msg))

