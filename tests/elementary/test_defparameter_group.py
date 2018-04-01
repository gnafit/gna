#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import load
import ROOT
import numpy as np
from gna.env import env
from gna.parameters.parameter_loader import get_parameters
from gna.config import cfg

# Necessary evil, it triggers import of all other symbols from shared library
ROOT.GNAObject

test_ns = env.ns("test")

pars = test_ns.defparameter_group(('par1', {'central': 1.0, 'relsigma': 0.1}),
                           ('par2', {'central': 2.0, 'relsigma': 0.1}),
                           **{'covmat': np.array([[1, 0.1], [0.1, 1]])})
import IPython
IPython.embed()
