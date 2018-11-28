#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import load
from gna.env import env

ns = env.globalns

ns.defparameter('par_fixed',           central=1.0,  fixed=True,   label='Fixed parameter')
ns.defparameter('par_free',            central=1.0,  free=True,    label='Free parameter')
ns.defparameter('par_constrained',     central=10.0, sigma=0.1,    label='Constrained parameter (absolute)')
ns.defparameter('par_constrained_rel', central=10.0, relsigma=0.1, label='Constrained parameter (relative)')
ns.printparameters(labels=True)

