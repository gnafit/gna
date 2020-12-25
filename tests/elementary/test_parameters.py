#!/usr/bin/env python
from load import ROOT
from gna.env import env
from gna.parameters.parameter_loader import get_parameters
from gna import parameters
parameters.debug=True

# Necessary evil, it triggers import of all other symbols from shared library
ROOT.GNAObject
Param_Double = ROOT.GaussianParameter('double')

env.defparameter('absolute sigma 1', central=0., sigma=1.)
env.defparameter('absolute sigma 2', central=2., sigma=0.1)
env.defparameter('relative sigma 1', central=0., relsigma=0.1)
env.defparameter('relative sigma 2', central=2., relsigma=0.1)

print()
env.defparameter('absolute sigma 3', central=2., uncertainty=0.1, uncertainty_type='absolute')
env.defparameter('relative sigma 3', central=2., uncertainty=0.1, uncertainty_type='relative')
gp=env.defparameter('absolute sigma with limits', central=2., relsigma=0.1, limits=[0.0, 3.0])
gp1=env.defparameter('absolute sigma with limits 1', central=2., relsigma=0.1, limits=[0.0, 3.0], fixed=True)
print()

env.defparameter('discrete', type='discrete', default='a', variants=dict(a=1, b=2, c=3))
env.defparameter('uniform angle', type='uniformangle', central=1.0)
env.defparameter('uniform angle fixed', type='uniformangle', central=2.0, fixed=True)
print()

env.defparameter('free', central=1.0, free=True)
env.defparameter('free with step', central=1.0, free=True, step=0.2)
env.defparameter('free 2', central=0.0, free=True)
env.defparameter('fixed 2', central=0.0, fixed=True)
print()

env.globalns.printparameters(labels=True)
