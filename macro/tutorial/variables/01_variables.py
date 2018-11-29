#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import load
from gna.env import env

# Make a variable for the global namespace
ns = env.globalns

# Create a set of parameters
ns.defparameter('par_constrained', central=10.0, sigma=0.1, label='Constrained parameter (absolute)')
ns.defparameter('par_constrained_rel', central=10.0, relsigma=0.1, label='Constrained parameter (relative)')
ns.defparameter('par_fixed', central=1.0, fixed=True, label='Fixed parameter')
ns.defparameter('par_free', central=1.0, free=True, label='Free parameter')
ns.reqparameter('par_free',central=2.0, free=True, label='Redefined free parameter')

# Print the list of the parameters to the terminal
ns.printparameters(labels=True)

