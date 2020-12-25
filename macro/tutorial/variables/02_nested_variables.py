#!/usr/bin/env python

import load
from gna.env import env

# Make a variable for global namespace
ns = env.globalns

# Create a parameter in the global namespace
ns.defparameter('norm', central=1.0, sigma=0.1, label='Normalization')

# Create (or access) nested namespace 'eff'
effns = ns('eff')

# Create a couple of parameters in the nested namespace
effns.defparameter('eff1', central=0.9, sigma=0.01, label='Efficiency #1')
effns.defparameter('eff2', central=0.8, sigma=0.02, label='Efficiency #2')

# Create a parameter in a nested namespace (from the global namespace)
ns.defparameter('eff.eff3', central=0.95, sigma=0.015, label='Efficiency #3')

# Create a couple of parameters in the third level namespace
ns.defparameter('misc.pars.par1', central=10.0, sigma=1, label='Some parameter')
ns.defparameter('misc.pars.par2', central=11.0, sigma=1, label='One more parameter')

# Print the list of parameters
ns.printparameters(labels=True)
