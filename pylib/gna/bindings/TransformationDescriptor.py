


from gna.bindings import patchROOTClass, DataType, provided_precisions
import ROOT as R
from printing import printl, nextlevel

classes = [R.TransformationDescriptorT(ft, ft) for ft in provided_precisions]

@patchROOTClass(classes, '__str__')
def TransformationDescriptor____str__(self):
    return '[trans] {}: {:d} input(s), {:d} output(s)'.format(self.name(), self.inputs.size(), self.outputs.size())

@patchROOTClass(classes, 'print')
def TransformationDescriptor__print(self, **kwargs):
    printl(str(self))
    with nextlevel():
        for i, inp in enumerate(self.inputs.values()):
            printl('{:2d}'.format(i), end=' ')
            inp.print(**kwargs)
        for i, o in enumerate(self.outputs.values()):
            printl('{:2d}'.format(i), end=' ')
            o.print(**kwargs)

@patchROOTClass(classes, 'single')
def TransformationDescriptor__single(self):
    outputs = self.outputs
    if outputs.size()!=1:
        raise Exception('Can not call single() on transformation %s with %i outputs'%(self.name(), outputs.size()))

    return outputs.front()

@patchROOTClass(classes, 'single_input')
def TransformationDescriptor__single_input(self):
    inputs = self.inputs
    if inputs.size()!=1:
        raise Exception('Can not call single_input() on transformation %s with %i inputs'%(self.name(), inputs.size()))

    return inputs.front()

@patchROOTClass(classes, '__rshift__')
def TransformationDescriptor______rshift__(transf, inputs):
    '''output(transf)>>inputs(arg)'''
    transf.single()>>inputs

@patchROOTClass(classes, '__rlshift__')
def TransformationDescriptor______rlshift__(transf, inputs):
    '''inputs(arg)<<output(transf)'''
    transf.single()>>inputs

@patchROOTClass(classes, '__lshift__')
def TransformationDescriptor______lshift__(self, output):
    '''inputs(self)<<output(arg)'''
    output>>self.single_input()

@patchROOTClass(classes, '__gt__')
@patchROOTClass(classes, '__lt__')
def TransformationDescriptor______cmp__(a, b):
    raise Exception('Someone tried to use >/< operators. Perhaps you have meant >>/<< instead?')
