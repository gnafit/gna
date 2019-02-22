#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import load
import ROOT
from gna.constructors import Points, stdvector 
from gna.ui import basecmd
import gna.parameters.oscillation
from gna.env import env
import gna.parameters

ROOT.GNAObject

ns = env.ns("004")
gna.parameters.oscillation.reqparameters(ns)

env.defparameter("L", central=810,sigma=0) #kilometre
env.defparameter("rho",central=2.7,sigma=0) #g/cm3

from_nu_a = ROOT.Neutrino.amu()
to_nu_a = ROOT.Neutrino.ae()

from_nu = ROOT.Neutrino.mu()
to_nu = ROOT.Neutrino.e()

E_arr = np.array(range(500, 6000, 50))  #array energy (МеV)

E = Points(E_arr)


with ns:
    #Vacuum neutrino (same antineutrino)
    oscprob = ROOT.OscProbPMNS(from_nu, to_nu)
    oscprob.full_osc_prob.inputs.Enu(E)
    data_osc = oscprob.full_osc_prob.oscprob




plt.plot(E_arr*1e-3, data_osc.data())
plt.xlabel('$E, GeV $')
plt.ylabel(r'$P_{\nu_{\mu} \to \nu_{e}}$')
plt.grid()
plt.show()
