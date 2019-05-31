#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import load
import ROOT
from gna.constructors import Points, stdvector,  OscProbPMNS
from gna import constructors as C # construct objects
from gna.ui import basecmd
import gna.parameters.oscillation
from gna.env import env
import gna.parameters
from gna import context, bindings

ROOT.GNAObject

ns = env.ns("004")

from_nu_a = ROOT.Neutrino.amu()
to_nu_a = ROOT.Neutrino.ae()

from_nu = ROOT.Neutrino.mu()
to_nu = ROOT.Neutrino.e()

E_arr = np.array(range(500, 720, 1))  #array energy (МеV)
#comp0 = np.array(np.ones(110))
E = Points(E_arr)
#com = Points(comp0)

ndata = 250

with context.manager(ndata) as manager:
  gna.parameters.oscillation.reqparameters(ns)

  env.defparameter("L", central=810,sigma=0) #kilometre
  env.defparameter("rho",central=2.7,sigma=0) #g/cm3

  with ns:
    #Vacuum neutrino (same antineutrino)
    oscprob = OscProbPMNS(from_nu, to_nu)
    oscprob.comp12.inputs.Enu(E)
#    oscprob.comp12.switchFunction("gpu")
    data_osc = oscprob.comp12.comp12

    oscprob.comp13.inputs.Enu(E)
#    oscprob.comp13.switchFunction("gpu")
    data_osc2 = oscprob.comp13.comp13

    oscprob.comp23.inputs.Enu(E)
 #   oscprob.comp23.switchFunction("gpu")
    data_osc3 = oscprob.comp23.comp23

    oscprob.compCP.inputs.Enu(E)
  #  oscprob.compCP.switchFunction("gpu")
    data_osc4 = oscprob.compCP.compCP


#    oscprob.probsum.inputs.comp12(oscprob.comp12.comp12)
#    oscprob.probsum.inputs.comp13(oscprob.comp13.comp13)
#    oscprob.probsum.inputs.comp23(oscprob.comp23.comp23)
#    oscprob.probsum.inputs.comp0(oscprob.compCP.compCP)
#    datafin = oscprob.probsum.probsum


#    oscprob = NestedDict(
#              bundle = dict(name='oscprob', version='v02', major='rdc'),
#              ),

#print(data_osc)
plt.plot(E_arr*1e-3, data_osc3.data(), label="comp23")
plt.plot(E_arr*1e-3, data_osc2.data(), label="comp13")
plt.plot(E_arr*1e-3, data_osc.data(), label="comp12")
plt.plot(E_arr*1e-3, data_osc4.data(), label="compCP")

oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()
oscprob.compCP.taint()


data_osc3.data()
data_osc2.data()
data_osc.data()
data_osc4.data()

oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()
oscprob.compCP.taint()

data_osc3.data()
data_osc2.data()
data_osc.data()
data_osc4.data()

oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()
oscprob.compCP.taint()

data_osc3.data()
data_osc2.data()
data_osc.data()
data_osc4.data()

oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()
oscprob.compCP.taint()

data_osc3.data()
data_osc2.data()
data_osc.data()
data_osc4.data()

oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()
oscprob.compCP.taint()

data_osc3.data()
data_osc2.data()
data_osc.data()
data_osc4.data()

oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()
oscprob.compCP.taint()

data_osc3.data()
data_osc2.data()
data_osc.data()
data_osc4.data()

oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()
oscprob.compCP.taint()

data_osc3.data()
data_osc2.data()
data_osc.data()
data_osc4.data()

oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()
oscprob.compCP.taint()

data_osc3.data()
data_osc2.data()
data_osc.data()
data_osc4.data()

oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()
oscprob.compCP.taint()

data_osc3.data()
data_osc2.data()
data_osc.data()
data_osc4.data()

oscprob.comp12.taint()
oscprob.comp13.taint()
oscprob.comp23.taint()
oscprob.compCP.taint()

data_osc3.data()
data_osc2.data()
data_osc.data()
data_osc4.data()


#plt.plot(E_arr*1e-3, datafin.data())
plt.xlabel('$comp_{ij}$')
plt.ylabel(r'$P_{\nu_{\mu} \to \nu_{e}}$')
plt.legend()
plt.grid()
plt.show()



