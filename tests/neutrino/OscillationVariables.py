#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
from gna.env import env

ns = env.globalns
ns.defparameter('L', central=1, sigma=0.1)

from gna.parameters.oscillation import reqparameters
reqparameters(ns)

op1 = R.OscProbPMNS(R.Neutrino.ae(), R.Neutrino.ae())
op2 = R.OscProbPMNS(R.Neutrino.ae(), R.Neutrino.ae())

ns.printparameters()

import IPython
IPython.embed()

