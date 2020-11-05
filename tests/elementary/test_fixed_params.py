#!/usr/bin/env python

import load
import ROOT
from gna.env import env
from gna.parameters.parameter_loader import get_parameters
# Necessary evil, it triggers import of all other symbols from shared library
ROOT.GNAObject

def test_fixed():
    env.defparameter('probe1', central=0., sigma=1.)
    env.defparameter('probe2', central=0., sigma=1.)
    env.defparameter('probe3', central=0., sigma=1.)
    env.defparameter('probe_fixed', central=0., sigma=1., fixed=True)

    print()
    msg = 'Testing that par {0} is created fixed' 
    print(msg.format(env.parameters['probe_fixed'].name()))
    assert env.parameters['probe_fixed'].isFixed() == True
    print('It is!\n')

    print("Checks whether get_parameters() discards fixed params by default:")
    no_fixed = [_.name() for _ in get_parameters(['probe1', 'probe_fixed'])]
    assert 'probe1' in no_fixed
    print('True!\n')

    print("Checks that get_parameters(drop_fixed=False) keeps parameters:")
    with_fixed =[par.name() for par in get_parameters(['probe1', 'probe_fixed'], drop_fixed=False)]
    assert 'probe1' in with_fixed and 'probe_fixed' in with_fixed
    print('True!\n')

if __name__ == "__main__":
    fixed_test()
