#!/usr/bin/env python
import load
import ROOT
from gna.env import env
from gna.parameters.parameter_loader import get_parameters

# Necessary evil, it triggers import of all other symbols from shared library
ROOT.GNAObject

def test_par_loader():
    probe1 = env.defparameter('probe1', central=0., sigma=1.)
    probe2 = env.defparameter('probe2', central=0., sigma=1.)
    probe3 = env.defparameter('probe3', central=0., sigma=1.)
    test_ns = env.ns('test_ns')
    test1 = test_ns.defparameter('test1', central=1., sigma=0.1)
    test2 = test_ns.defparameter('test2', central=1., sigma=0.1)
    test3 = test_ns.defparameter('test3', central=1., sigma=0.1)
    test4 = test_ns.defparameter('test4', central=1., sigma=0.1)
    extra_test_ns = env.ns('extra_test_ns')
    extra1 =extra_test_ns.defparameter('extra1', central=1., sigma=0.1)

    for name, par in test_ns.walknames():
        print("Par in namespace", name)

    print()
    print("Quering pars with get_parameters() ")
    par1 = get_parameters(['probe1'])
    assert(par1[0] == probe1)
    print('Got global parameter {}'.format(par1[0].name()))
    print()
    par2 = get_parameters(['test_ns.test1'])
    assert(par2[0] == test1)
    print('Got parameter {0} from namespace {1}'.format(par2[0].name(),
        test_ns.name))

    print()

    par_in_namespace = get_parameters(['test_ns'])
    assert(par_in_namespace == [test1, test2, test3, test4]) 
    print()
    print('Got parameters {0} from ns {ns}'.format([_.name() for _ in
        par_in_namespace], ns=test_ns.name))
    print()
    par_two_namespaces = get_parameters(['test_ns', 'extra_test_ns'])
    assert(par_two_namespaces == [test1, test2, test3, test4, extra1])
    print()
    print('Got parameters {0} from nses {ns}'.format([_.name() for _ in
           par_two_namespaces], ns=[test_ns.name, extra_test_ns.name]))
    print()

    par_mixed_ns_and_global = get_parameters(['test_ns', 'probe1'])
    print('Got parameters {0} from nses {ns} and global'.format([_.name() for _ in
           par_mixed_ns_and_global], ns=test_ns.name))
    assert(par_mixed_ns_and_global == [test1, test2, test3, test4, probe1])

    print()

    wildcard = get_parameters(['test_ns*'])
    assert(wildcard == [test1, test2, test3, test4])
    print('Got parameters {0} from by wildcard '.format([_.name() for _ in
           wildcard] ))

    print()

    #Asking for missing parameters, raise KeyError
    try:
        get_parameters(['missing'])
    except KeyError:
        print("KeyError for missing parameter is raised correctly")

if __name__ == "__main__":
    test_par_loader()
