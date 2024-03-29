#!/usr/bin/env python

from gna.expression.simple import *
import colorama

class IndexedContainer(object):
    objects = None
    operator='.'
    text_operator='_period_'
    left, right = '', ''
    prefix = ''
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

    def set_operator(self, operator, left=None, right=None, text=None, prefix=None):
        self.operator=operator
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right
        if text is not None:
            self.text_operator=text
        if prefix is not None:
            self.prefix = prefix

    def guessname(self, lib={}, save=False):
        for o in self.objects:
            o.guessname(lib, save)

        if self.name is not undefinedname:
            if not self.label:
                for cfg in lib.values():
                    if cfg['name']!=self.name:
                        continue
                    newlabel = cfg.get('label', None)
                    if newlabel:
                        self.label=newlabel
                    self.expandable = cfg.get('expand', self.expandable)
            return self.name

        newname = '{prefix}{expr}'.format( prefix=self.prefix,
                    expr = self.operator.strip().join(sorted(o.ident(lib=lib, save=save) for o in self.objects)),
                    )

        newnameu = '{prefix}{expr}'.format( prefix=self.prefix,
                    expr = self.operator.strip().join(o.ident(lib=lib, save=save) for o in self.objects),
                     )

        variants=[newnameu, newname]
        for nn in tuple(variants):
            variants.append(nn+':'+self.nindex.ident())

        label = None
        for var in variants:
            if var in lib:
                libentry = lib[var]
                newname = libentry['name']
                label   = libentry.get('label', None)
                break
        else:
            newname = '{expr}'.format(
                        expr = self.text_operator.strip().join(sorted(o.ident(lib=lib, save=save) for o in self.objects)),
                        )

        if save:
            self.name = newname
            self.set_label(label)

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

    @methodname
    def require(self, context):
        for obj in self.objects:
            obj.require(context)

    @call_once
    def bind(self, context, connect=True):
        if not self.objects:
            return

        printl_debug('bind (container) {}:'.format(type(self).__name__), str(self) )

        with nextlevel():
            for obj in self.objects:
                obj.bind(context)

            if not connect:
                return

            printl_debug('connect (container)', str(self))
            with nextlevel():
                for idx in self.nindex.iterate():
                    printl_debug( 'index', idx )
                    with nextlevel():
                        nobj = len(self.objects)
                        for i, obj in enumerate(self.objects):
                            # if nobj==1:
                                # i=None
                            output = obj.get_output(idx, context)
                            inputs  = self.get_input(idx, context, clone=i)

                            if not isinstance(inputs, (list, tuple)):
                                inputs = inputs,
                            for input in inputs:
                                if not input.materialized(): #Fixme: should be configurable

                                    try:
                                        output >> input
                                    except:
                                        print(colorama.Fore.RED+f'Unable to bind {self.name}: {output.name()} >> {input.name()}'+colorama.Style.NORMAL)
                                        raise

class VProduct(IndexedContainer, Variable):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for VarProduct')

        newobjects = []
        for o in objects:
            if not isinstance(o, Variable):
                raise Exception('Expect Variable instance')

            if self.expandable and isinstance(o, VProduct) and o.expandable:
                newobjects+=o.objects
            else:
                newobjects.append(o)

        IndexedContainer.__init__(self, *newobjects)
        Variable.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( '*', text='_times_' )

    @call_once
    def bind(self, context):
        printl_debug('bind (var) {}:'.format(type(self).__name__), str(self) )
        from gna.env import ExpressionsEntry, ExpressionWithBindings
        with nextlevel():
            IndexedContainer.bind(self, context, connect=False)

            from gna.constructors import VarProduct
            with context.ns:
                for idx in self.nindex.iterate():
                    names = [obj.current_format(idx) for obj in self.objects]
                    name = self.current_format(idx)

                    if '.' in name:
                        path, head = name.rsplit('.', 1)
                        ns = context.ns(path)
                    else:
                        path, head = '', name
                        ns = context.ns
                    vp = VarProduct(names, head, ns=ns)
                    v=ns[head]
                    if isinstance(v, (ExpressionsEntry, ExpressionWithBindings)):
                        v=v.get()
                    v.setLabel( name+' = '+' * '.join(names) )

class NestedTransformation(object):
    tinit = None

    def __init__(self):
        self.tobjects = []

    def set_tinit(self, obj):
        self.tinit = obj

    def new_tobject(self, idx, *args, **kwargs):
        if isinstance(idx, str):
            label = idx
        elif self.label is not None:
            label = idx.current_format(self.label, name=self.name, **kwargs)
        else:
            label = self.current_format(idx, **kwargs)

        newobj = self.tinit(*args)
        newobj.transformations[0].setLabel(label)
        self.tobjects.append(newobj)
        import ROOT as R
        return newobj, R.OutputDescriptor(newobj.single())

    @call_once
    def bind(self, context):
        printl_debug('bind (nested) {}:'.format(type(self).__name__), str(self) )

        if self.tinit:
            with nextlevel():
                for idx in self.nindex.iterate():
                    tobj, newout = self.new_tobject(idx)
                    context.set_output(newout, self.name, idx)
                    nobj = len(self.objects)
                    for i, obj in enumerate(self.objects):
                        # if nobj==1:
                            # i=None
                        inp = self.add_input(tobj, i)
                        context.set_input(inp, self.name, idx, clone=i)

        with nextlevel():
            IndexedContainer.bind(self, context)

    def add_input(self, tobj, idx):
        return tobj.add_input('%02d'%idx)

    @methodname
    def require(self, context):
        IndexedContainer.require(self, context)

class TCall(IndexedContainer, Transformation):
    inputs_connected = False
    def __init__(self, name, *args, **kwargs):
        targs = ()
        if '|' in args:
            idx = args.index('|')
            args, targs = args[:idx], args[idx+1:]

        targs = list(targs) + list(kwargs.pop('targs', ()))

        objects = []
        for iarg, arg in enumerate(targs):
            if isinstance(arg, str):
                arg = Transformation(arg)
            elif not isinstance(arg, Transformation):
                raise Exception(f'Argument {iarg} ({arg.name}) of {name} should be another Transformation')
            objects.append(arg)

        IndexedContainer.__init__(self, *objects)
        Transformation.__init__(self, name, *(list(args)+list(objects)), **kwargs)
        self.set_operator( ', ', '(', ')', text='_and_'  )

        self.inputs_connected = not self.nonempty()

    def __str__(self):
        return '{}({:s})'.format(Indexed.__str__(self), '...' if self.objects else '' )

    def estr(self, expand=100):
        if expand:
            expand-=1
            return '{fcn}{args}'.format(fcn=Indexed.__str__(self), args=IndexedContainer.estr(self, expand))
        else:
            return self.__str__()

    @call_once
    def bind(self, context):
        if self.inputs_connected:
            return

        printl_debug('bind (call) {}:'.format(type(self).__name__), str(self) )
        with nextlevel():
            IndexedContainer.bind(self, context)

        self.inputs_connected = True

    @methodname
    def require(self, context):
        Transformation.require(self, context)
        IndexedContainer.require(self, context)

class TProduct(NestedTransformation, IndexedContainer, Transformation):
    def __init__(self, name, *objects, **kwargs):
        import ROOT as R
        # pclass = R.GNA.GNAObjectTemplates.ProductT('double')
        pclass = R.GNA.GNAObjectTemplates.ProductBCT('double')

        if not objects:
            raise Exception('Expect at least one variable for TProduct')

        newobjects = []
        for o in objects:
            if not isinstance(o, Transformation):
                raise Exception('Expect Transformation instance, got {} ({})'.format(
                    hasattr(o, 'name') and o.name or '', type(o).__name__
                    ))

            if self.expandable and isinstance(o, TProduct) and o.expandable and not o is self:
                newobjects+=o.objects
            else:
                newobjects.append(o)

        NestedTransformation.__init__(self)
        IndexedContainer.__init__(self, *newobjects)
        Transformation.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( ' * ', '( ', ' )', text='_times_'  )
        self.set_tinit( pclass )

class TRatio(NestedTransformation, IndexedContainer, Transformation):
    def __init__(self, name, *objects, **kwargs):
        bc = kwargs.pop('broadcast', False)
        if len(objects)!=2:
            raise Exception('Expect two objects for TRatio')

        for o in objects:
            if not isinstance(o, Transformation):
                raise Exception('Expect Transformation instance, got {} ({})'.format(
                    hasattr(o, 'name') and o.name or '', type(o).__name__
                    ))

        NestedTransformation.__init__(self)
        IndexedContainer.__init__(self, *objects)
        Transformation.__init__(self, name, *objects, **kwargs)

        self.set_operator( ' / ', '( ', ' )', text='_over_'  )
        import ROOT as R
        self.set_tinit(bc and R.RatioBC or R.Ratio)

    def add_input(self, tobj, idx):
        if not idx in [0, 1]:
            raise Exception('Ratio argument indices sould be 0, 1, but not '+str(idx))
        return tobj.ratio.inputs[idx]

class TSum(NestedTransformation, IndexedContainer, Transformation):
    def __init__(self, name, *objects, **kwargs):
        if not objects:
            raise Exception('Expect at least one variable for TSum')

        newobjects = []
        for o in objects:
            if not isinstance(o, Transformation):
                raise Exception('Expect Transformation instance, got {} ({})'.format(
                    hasattr(o, 'name') and o.name or '', type(o).__name__
                    ))

            if self.expandable and isinstance(o, TSum) and o.expandable and not o is self:
                newobjects+=o.objects
            else:
                newobjects.append(o)

        NestedTransformation.__init__(self)
        IndexedContainer.__init__(self, *newobjects)
        Transformation.__init__(self, name, *newobjects, **kwargs)

        self.set_operator( ' + ', '( ', ' )', text='_plus_' )
        import ROOT as R
        self.set_tinit( R.Sum )

class WeightedTransformation(NestedTransformation, IndexedContainer, Transformation):
    object, weight = None, None
    def __init__(self, name, *objects, **kwargs):
        for other in objects:
            if self.expandable and isinstance(other, WeightedTransformation) and other.expandable:
                self.object = self.object*other.object if self.object is not None else other.object
                self.weight = self.weight*other.weight if self.weight is not None else other.weight
            elif isinstance(other, Variable):
                self.weight = self.weight*other if self.weight is not None else other
            elif isinstance(other, Transformation):
                self.object = Transformation.__mul__(self.object, other, False) if self.object is not None else other
            else:
                raise Exception( 'Unsupported type' )

        NestedTransformation.__init__(self)
        IndexedContainer.__init__(self, self.weight, self.object)
        Transformation.__init__(self, name, self.weight, self.object, **kwargs)

        self.set_operator( ' * ', text='_times_'  )

        import ROOT as R
        self.set_tinit( R.WeightedSum )

    @methodname
    def require(self, context):
        self.object.require(context)
        self.weight.require(context)

    @call_once
    def bind(self, context):
        printl_debug('bind (weighted) {}:'.format(type(self).__name__), str(self) )
        with nextlevel():
            IndexedContainer.bind(self, context, connect=False)

            from gna.constructors import stdvector
            if self.object.name is undefinedname:
                raise Exception('May not work with objects with undefined names')
            labels  = stdvector([self.object.name])
            printl_debug('connect (weighted)')
            for idx in self.nindex.iterate():
                wname = self.weight.current_format(idx)
                weights = stdvector([wname])

                with context.ns:
                    tobj, newout = self.new_tobject(idx, weights, labels, weight_label=self.weight.name)
                inp = tobj.transformations[0].inputs[0]
                context.set_output(newout, self.name, idx)
                context.set_input(inp, self.name, idx)
                out = self.object.get_output(idx, context)
                out >> inp

    def test_iteration(self):
        for it in self.nindex.iterate():
            print('index', it.current_format())
            print('  weight', self.weight.current_format(it))
            print('  obj', self.object.current_format(it))
