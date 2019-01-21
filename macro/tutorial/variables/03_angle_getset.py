#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import load
from gna.env import env
import numpy as np

# Make a variable for global namespace
ns = env.globalns

# Create a parameter in the global namespace
norm = ns.defparameter('norm', central=1.0, sigma=0.1, label='Normalization')
ns.defparameter('phase', central=np.pi*0.5, type='uniformangle', label='Phase angle')

phase = ns['phase']

# Print the list of parameters
ns.printparameters(labels=True)
print()

# Change parameters 1
norm.set(2.0)
phase.set(6.28)
ns.printparameters(labels=True)
print()

# Change parameters 2
norm.setNormalValue(1.0)
phase.set(-3.13)
ns.printparameters(labels=True)
print()

# Get parameters' values
norm_value = norm.value()
phase_value = phase.value()
print('Norm', norm_value)
print('Phase', phase_value)
