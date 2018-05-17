# -*- coding: utf-8 -*-

from __future__ import print_function
from collections import OrderedDict
from gna.configurator import NestedDict
import itertools as I

printlevel = 0
class nextlevel():
    def __enter__(self):
        global printlevel
        printlevel+=1

    def __exit__(self, *args, **kwargs):
        global printlevel
        printlevel-=1

def printl(*args, **kwargs):
    prefix = kwargs.pop('prefix', ())

    if prefix:
        print( *prefix, end='' )
    print('    '*printlevel, sep='', end='')
    print(*args, **kwargs)

class Index(object):
    def __init__(self, short, name, variants):
        self.short = short
        self.name  = name
        self.variants = variants

    def iterate(self, mode='values', fix={}):
        if self.variants is None:
            raise Exception( 'Variants are not initialized for {name}'.format(**self.__dict__) )

        val = fix.get(self.name, fix.get(self.short, None))
        if val is not None:
            if not val in self.variants:
                raise Exception( 'Can not fix index {name} in value {value}. Variants are: {variants:s}'.format( **self.__dict__ ) )
            variants = val,
        else:
            variants = self.variants

        if mode=='values':
            for var in variants:
                yield var
        elif mode=='items':
            for var in variants:
                yield self.short, var
        elif mode=='longitems':
            for var in variants:
                yield self.name, var
        # elif mode=='bothitems':
            # for var in variants:
                # yield self.short, self.name, var
        else:
            raise Exception('Unsupported iteration mode: {}'.format(mode))

    __iter__ = iterate

    def __str__(self):
        return '{name} ({short}): {variants:s}'.format( **self.__dict__ )

class NIndex(object):
    def __init__(self, *indices, **kwargs):
        self.indices = OrderedDict()

        for idx in indices:
            self |= idx

        ignore = kwargs.pop('ignore', None)
        if ignore:
            for other in ignore:
                if other in self.indices:
                    del self.indices[other]

        fromlist = kwargs.pop('fromlist', [])
        for args in fromlist:
            idx = Index(*args)
            self |= idx

        self.arrange()

        if kwargs:
            raise Exception('Unparsed kwargs: {:s}'.format(kwargs))

    def __ior__(self, other):
        if isinstance(other, Index):
            self.indices[other.short]=other
        elif isinstance(other, str):
            self.indices[other]=Index(other, other, variants=None)
        else:
            if isinstance(other, NIndex):
                others = other.indices.values()
            elif isinstance(other, Indexed):
                others = other.indices.indices.values()
            else:
                raise Exception( 'Unsupported index type '+type(other).__name__ )

            for other in others:
                self.indices[other.short]=other

        return self

    def __sub__(self, other):
        return NIndex(self, ignore=other.indices.keys())

    def arrange(self):
        self.indices = OrderedDict( [(k, self.indices[k]) for k in sorted(self.indices.keys())] )

    def __str__(self):
        return ', '.join( self.indices.keys() )

    def __add__(self, other):
        return NIndex(self, other)

    def __bool__(self):
        return bool(self.indices)

    __nonzero__ = __bool__

    def __eq__(self, other):
        if not isinstance(other, NIndex):
            return False
        return self.indices==other.indices

    # def reduce(self, *indices):
        # if not set(indices.keys()).issubset(self.indices.keys()):
            # raise Exception( "NIndex.reduce should be called on a subset of indices, got {:s} in {:s}".format(indices.keys(), self.indices.keys()) )

        # return NIndex(*(set(self.indices)-set(indices))) #legacy

    def ident(self, **kwargs):
        return '_'.join(self.indices.keys())

    def names(self, short=False):
        if short:
            return [idx.short for idx in self.indices.values()]
        else:
            return [idx.name for idx in self.indices.values()]

    def iterate(self, mode='values', fix={}, **kwargs):
        autoindex = '{'+'}.{'.join(self.names())+'}'
        if autoindex!='{}':
            autoindex = '.'+autoindex

        if mode in ['values']:
            for it in I.product(*(idx.iterate(mode=mode, fix=fix) for idx in self.indices.values())):
                yield it
        elif mode in ['items']:
            for it in I.product(*(idx.iterate(mode=mode, fix=fix) for idx in self.indices.values())):
                yield it
        elif mode in ['longitems']:
            for it in I.product(*(idx.iterate(mode=mode, fix=fix) for idx in self.indices.values())):
                res = it + tuple(kwargs.items())
                res = res + (( 'autoindex', autoindex.format('', **dict(res))),)
                yield res
        else:
            if not isinstance(mode, str):
                raise Exception('Mode should be a string, got {}'.format(type(str).__name__))

            for it in self.iterate(mode='longitems', fix=fix, **kwargs):
                dct = OrderedDict(it)
                yield mode.format(**dct)

    __iter__ = iterate

    def get_relevant(self, nidx):
        return type(nidx)([(k,v) for (k,v) in nidx if k in self.indices])

class Indexed(object):
    name=''
    indices_locked=False
    fmt=None
    def __init__(self, name, *indices, **kwargs):
        self.name=name
        self.set_indices(*indices, **kwargs)

    def set_format(self, fmt):
        self.fmt = fmt

    def set_indices(self, *indices, **kwargs):
        self.indices=NIndex(*indices, **kwargs)
        if indices:
            self.indices_locked=True

    def __getitem__(self, args):
        if self.indices_locked:
            raise Exception('May not modify already declared indices')
        if not isinstance(args, tuple):
            args = args,
        self.set_indices(*args)
        return self

    def __add__(self, other):
        raise Exception('not implemented')

    def __str__(self):
        if self.indices:
            return '{}[{}]'.format(self.name, str(self.indices))
        else:
            return self.name

    def estr(self, expand=100):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Indexed):
            return False
        if self.name!=other.name:
            return False
        return self.indices==other.indices

    # def reduce(self, newname, *indices):
        # return Indexed(newname, self.indices.reduce(*indices))

    def walk(self, yieldself=False, operation=''):
        yield self, operation

    def ident(self, **kwargs):
        if self.name=='?':
            return self.guessname(**kwargs)
        return self.name

    def ident_full(self, **kwargs):
        return '{}:{}'.format(self.ident(**kwargs), self.indices.ident(**kwargs))

    def guessname(self, *args, **kwargs):
        return '?'

    def dump(self, yieldself=False):
        for i, (obj, operator) in enumerate(self.walk(yieldself)):
            printl(operator, obj, prefix=(i, printlevel) )

    def get_output(self, nidx, context):
        return context.get_output(self.name, self.get_relevant( nidx ))

    def get_input(self, nidx, context, clone=None):
        return context.get_input(self.name, self.get_relevant( nidx ), clone=clone)

    def get_relevant(self, nidx):
        return self.indices.get_relevant(nidx)

class IndexedContainer(object):
    objects = None
    operator='.'
    left, right = '', ''
    def __init__(self, *objects):
        self.set_objects(*objects)

    def set_objects(self, *objects):
        self.objects = list(objects)

    def walk(self, yieldself=False, operation=''):
        if yieldself or not self.objects:
            yield self, operation
        with nextlevel():
            for o in self.objects:
                for sub in  o.walk(yieldself, self.operator.strip()):
                    yield sub

    def set_operator(self, operator, left=None, right=None):
        self.operator=operator
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right

    def guessname(self, lib={}, save=False):
        for o in self.objects:
            o.guessname(lib, save)

        newname = '{expr}'.format(
                    expr = self.operator.strip().join(sorted(o.ident(lib=lib, save=save) for o in self.objects)),
                    )

        newnameu = '{expr}'.format(
                    expr = self.operator.strip().join(o.ident(lib=lib, save=save) for o in self.objects),
                     )

        variants=[newnameu, newname]
        for nn in tuple(variants):
            variants.append(nn+':'+self.indices.ident())

        guessed = False
        for var in variants:
            if var in lib:
                guessed = lib[var]['name']
                break

        if guessed:
            newname = guessed
            if save:
                self.name = newname

        return newname

    def estr(self, expand=100):
        if expand:
            expand = expand-1
            return '{left}{expr}{right}'.format(
                    left = self.left,
                    expr = self.operator.join(o.estr(expand) for o in self.objects),
                    right= self.right
                    )
        else:
            return self.__str__()

    def nonempty(self):
        return bool(self.objects)

    def build(self, context, connect=True):
        if not self.objects:
            return

        printl('build (container) {}:'.format(type(self).__name__), str(self) )

        with nextlevel():
            for obj in self.objects:
                context.check_outputs(obj)

            if not connect:
                return

            printl('connect (container)', str(self))
            with nextlevel():
                for idx in self.indices.iterate(mode='items'):
                    printl( 'index', idx )
                    with nextlevel():
                        for i, obj in enumerate(self.objects):
                            output = obj.get_output(idx, context)
                            input  = self.get_input(idx, context, clone=i)
                            input(output)

class Variable(Indexed):
    def __init__(self, name, *args, **kwargs):
        super(Variable, self).__init__(name, *args, **kwargs)

    def __mul__(self, other):
        if isinstance(other, Transformation):
            return WeightedTransformation('?', self, other)

        return VProduct('?', self, other)

    def __call__(self, *targs):
        return TCall(self.name, self, targs=targs)

    def build(self, context):
        pass

    def get_output(self, nidx, context):
        pass

class VProduct(IndexedContainer, Variable):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for VarProduct')

        newobjects = []
        for o in objects:
            if not isinstance(o, Variable):
                raise Exception('Expect Variable instance')

            if isinstance(o, VProduct):
                newobjects+=o.objects
            else:
                newobjects.append(o)

        IndexedContainer.__init__(self, *newobjects)
        Variable.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( '*' )

class Transformation(Indexed):
    def __init__(self, name, *args, **kwargs):
        super(Transformation, self).__init__(name, *args, **kwargs)

    def __str__(self):
        return '{}()'.format(Indexed.__str__(self))

    def __mul__(self, other):
        if isinstance(other, (Variable, WeightedTransformation)):
            return WeightedTransformation('?', self, other)

        return TProduct('?', self, other)

    def __add__(self, other):
        return TSum('?', self, other)

    def build(self, context):
        printl('build (trans) {}:'.format(type(self).__name__), str(self) )
        context.build( self.name, self.indices)

class NestedTransformation(object):
    tinit = None

    def __init__(self):
        self.tobjects = []

    def set_tinit(self, obj):
        self.tinit = obj

    def new_tobject(self, label):
        newobj = self.tinit()
        newobj.transformations[0].setLabel(label)
        self.tobjects.append(newobj)
        import ROOT as R
        return newobj, R.OutputDescriptor(newobj.single())

    def build(self, context):
        printl('build (nested) {}:'.format(type(self).__name__), str(self) )

        if self.tinit:
            with nextlevel():
                for idx in self.indices.iterate(mode='items'):
                    tobj, newout = self.new_tobject(str(idx))
                    context.set_output(newout, self.name, idx)
                    for i, obj in enumerate(self.objects):
                        inp = tobj.add_input('%02d'%i)
                        context.set_input(inp, self.name, idx, clone=i)

        with nextlevel():
            IndexedContainer.build(self, context)

class TCall(IndexedContainer, Transformation):
    def __init__(self, name, *args, **kwargs):
        targs = ()
        if '|' in args:
            idx = args.index('|')
            args, targs = args[:idx], args[idx+1:]

        targs = list(targs) + list(kwargs.pop('targs', ()))

        objects = []
        for arg in targs:
            if isinstance(arg, str):
                arg = Transformation(arg)
            elif not isinstance(arg, Transformation):
                raise Exception('Arguments argument should be another Transformation')
            objects.append(arg)

        IndexedContainer.__init__(self, *objects)
        Transformation.__init__(self,name, *(list(args)+list(objects)), **kwargs)
        self.set_operator( ', ', '(', ')' )

    def __str__(self):
        return '{}({:s})'.format(Indexed.__str__(self), '...' if self.objects else '' )

    def estr(self, expand=100):
        if expand:
            expand-=1
            return '{fcn}{args}'.format(fcn=Indexed.__str__(self), args=IndexedContainer.estr(self, expand))
        else:
            return self.__str__()

    def build(self, context):
        printl('build (call) {}:'.format(type(self).__name__), str(self) )
        with nextlevel():
            Transformation.build(self, context)
            IndexedContainer.build(self, context)

class TProduct(NestedTransformation, IndexedContainer, Transformation):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for TProduct')

        newobjects = []
        for o in objects:
            if not isinstance(o, Transformation):
                raise Exception('Expect Transformation instance')

            if isinstance(o, TProduct):
                newobjects+=o.objects
            else:
                newobjects.append(o)

        NestedTransformation.__init__(self)
        IndexedContainer.__init__(self, *newobjects)
        Transformation.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( ' * ' )
        import ROOT as R
        self.set_tinit( R.Product )

class TSum(NestedTransformation, IndexedContainer, Transformation):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for TSum')

        newobjects = []
        for o in objects:
            if not isinstance(o, Transformation):
                raise Exception('Expect Transformation instance')

            if isinstance(o, TSum):
                newobjects+=o.objects
            else:
                newobjects.append(o)

        NestedTransformation.__init__(self)
        IndexedContainer.__init__(self, *newobjects)
        Transformation.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( ' + ', '(', ')' )
        import ROOT as R
        self.set_tinit( R.Sum )

class WeightedTransformation(NestedTransformation, IndexedContainer, Transformation):
    object, weight = None, None
    def __init__(self, name, *objects, **kwargs):
        for other in objects:
            if isinstance(other, WeightedTransformation):
                self.object = self.object*other.object if self.object is not None else other.object
                self.weight = self.weight*other.weight if self.weight is not None else other.weight
            elif isinstance(other, Variable):
                self.weight = self.weight*other if self.weight is not None else other
            elif isinstance(other, Transformation):
                self.object = self.object*other if self.object is not None else other
            else:
                raise Exception( 'Unsupported type' )

        NestedTransformation.__init__(self)
        IndexedContainer.__init__(self, self.weight, self.object)
        Transformation.__init__(self, name, self.weight, self.object, **kwargs)

        self.set_operator( ' * ' )

    def __mul__(self, other):
        return WeightedTransformation('?', self, other)

class OperationMeta(type):
    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = args,
        return cls(*args)

class Operation(TCall,NestedTransformation):
    __metaclass__ = OperationMeta
    call_lock=False
    def __init__(self, name, *indices, **kwargs):
        self.indices_to_reduce = NIndex(*indices)
        TCall.__init__(self, name)
        NestedTransformation.__init__(self)

    def __str__(self):
        return '{}{{{:s}}}'.format(Indexed.__str__(self), self.indices_to_reduce)

    def __call__(self, *args):
        if self.call_lock:
            raise Exception('May call Operation only once')
        self.call_lock=True

        self.set_objects(*args)
        self.set_indices(*args, ignore=self.indices_to_reduce.indices.keys())
        return self

    def guessname(self, lib={}, save=False):
        for o in self.objects:
            o.guessname(lib, save)

        newname=self.name+':'+self.indices_to_reduce.ident()
        if newname in lib:
            newname = lib[newname]['name']

            if save:
                self.name = newname
        return newname

    def make_object(self, *args, **kwargs):
        raise Exception('Unimplemented method called')

    def build(self, context):
        printl('build (operation) {}:'.format(type(self).__name__), str(self) )

        with nextlevel():
            IndexedContainer.build(self, context, connect=False)

        if not self.tinit:
            return

        with nextlevel():
            printl('connect (operation)', str(self))
            with nextlevel():
                for freeidx in self.indices.iterate(mode='items'):
                    tobj, newout = self.new_tobject(str(freeidx))
                    context.set_output(newout, self.name, freeidx)
                    with nextlevel():
                        for opidx in self.indices_to_reduce.iterate(mode='items'):
                            fullidx = list(freeidx)+list(opidx)
                            for i, obj in enumerate(self.objects):
                                output = obj.get_output(fullidx, context)
                                inp  = tobj.add_input('%02d'%i)
                                context.set_input(inp, self.name, fullidx, clone=i)
                                inp(output)

class OSum(Operation):
    def __init__(self, *indices, **kwargs):
        Operation.__init__(self, 'sum', *indices, **kwargs)
        self.set_operator( ' ++ ' )

        import ROOT as R
        self.set_tinit( R.Sum )

placeholder=['placeholder']
class OProd(Operation):
    def __init__(self, *indices, **kwargs):
        Operation.__init__(self, 'prod', *indices, **kwargs)
        self.set_operator( ' ** ' )

        import ROOT as R
        self.set_tinit( R.Product )

class VTContainer(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(VTContainer, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        newvar = Variable(key)
        self[key] = newvar
        return newvar

class Expression(object):
    operations = dict(sum=OSum, prod=OProd)
    tree = None
    def __init__(self, expression, indices):
        self.expression_raw = expression
        self.expression = self.preprocess( self.expression_raw )

        self.globals=VTContainer(self.operations)
        self.indices=OrderedDict()
        self.defindices(indices)

    def preprocess(self, expression):
        n = expression.count('|')
        if n:
            expression = expression.replace('|', '(')
            expression+=')'*n
        return expression

    def parse(self):
        if self.tree:
            raise Exception('Expression is already parsed')
        self.tree=eval(self.expression, self.globals)
        self.guessname=self.tree.guessname

    def __str__(self):
        return self.expression_raw

    def __repr__(self):
        return 'Expression("{}")'.format(self.expression_raw)

    def newindex(self, short, name, variants):
        idx = self.indices[short] = Index(short, name, variants)
        self.globals[short] = idx

    def defindices(self, defs):
        for d in defs:
            self.newindex(*d)

    def build(self, context):
        if not self.tree:
            raise Exception('Expression is not initialized, call parse() method first')

        context.set_indices(self.indices)
        self.tree.build( context )

class ExpressionContext(object):
    indices = None
    def __init__(self, cfg):
        self.cfg = cfg
        self.outputs = NestedDict()
        self.inputs  = NestedDict()

        self.providers = dict()
        for keys, value in cfg.items():
            if isinstance(value, NestedDict) and 'provides' in value:
                keys=value.provides

            if not isinstance(keys, (list, tuple)):
                keys=keys,

            for key in keys:
                self.providers[key]=value

    def set_indices(self, indices):
        self.indices = indices

    def build(self, name, indices):
        cfg = self.providers.get(name, None)
        if cfg is None:
            if indices:
                fmt='{name}{autoindex}' #FIXME: hardcoded
                for it in indices.iterate( mode=fmt, name=name ):
                    self.build(it, None)
                return

            raise Exception('Do not know how to build '+name)

        printl('build', name, 'via bundle' )

        if indices is not None:
            cfg.indices=indices

        from gna.bundle import execute_bundle
        with nextlevel():
            b=execute_bundle( cfg=cfg, context=self )

    def get_variable(self, name, *idx):
        pass

    def check_outputs(self, obj):
        printl( 'check', obj )
        with nextlevel():
            if obj.name in self.outputs:
                printl( 'found' )
                return

            obj.build(self)

    def get_key(self, name, nidx, fmt=None, clone=None):
        if clone is not None:
            clone = '%02d'%clone

        nidx = OrderedDict(nidx)
        if fmt:
            ret = fmt.format( **nidx )
            if clone:
                ret += '.'+clone
            return ret

        nidx = tuple([nidx[k] for k in sorted(nidx.keys())])
        if clone:
            nidx = nidx + (clone,)
        return (name,)+nidx

    def get_output(self, name, nidx):
        return self.get( self.outputs, name, nidx, 'output' )

    def set_output(self, output, name, nidx, fmt=None, **kwargs):
        self.set( self.outputs, output, name, nidx, 'output', fmt, **kwargs )

    def get_input(self, name, nidx, clone=None):
        return self.get( self.inputs, name, nidx, 'input', clone=clone )

    def set_input(self, input, name, nidx, fmt=None, clone=None):
        self.set( self.inputs, input, name, nidx, 'input', fmt, clone)

    def get(self, source, name, nidx, type, clone=None):
        key = self.get_key(name, nidx, clone=clone)
        printl('get {}'.format(type), name, key)

        ret = source.get(key, None)
        if not ret:
            raise Exception('Failed to get {} {}[{}]'.format(type, name, nidx, clone))

        return ret

    def set(self, target, io, name, nidx, type, fmt=None, clone=None):
        key = self.get_key( name, nidx, fmt, clone )
        printl('set {}'.format(type), name, key)
        target[key]=io

    # def connect(self, source, sink, nidx, fmtsource=None, fmtsink=None):
        # printl( 'connect: {}->{} ({:s})'.format( source, sink, nidx ) )
        # with nextlevel():
            # output = self.get_output( source, nidx )
            # input  = self.get_input( sink, nidx )

        # input( output )
