#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import ROOT
from gna.ui import basecmd
import gna.parameters.oscillation

self.ns = self.env.ns("004")
gna.parameters.oscillation.reqparameters(self.ns)
self.ns.defparameter("L", central=810,sigma=0) #kilometre
self.ns.defparameter("rho",central=2.7,sigma=0) #g/cm3

from_nu_a = ROOT.Neutrino.amu()
to_nu_a = ROOT.Neutrino.ae()

from_nu = ROOT.Neutrino.mu()
to_nu = ROOT.Neutrino.e()

E_arr = np.array(range(500, 6000, 50))

E = ROOT.Points(E_arr)


with self.ns:
    #Matter neutrino
    oscprob_m = ROOT.OscProbMatter(from_nu, to_nu)
    oscprob_m.oscprob.Enu(E)
    data_osc_m = oscprob_m.oscprob.oscprob

    #Matter antineutrino
    oscprob_ma = ROOT.OscProbMatter(from_nu_a, to_nu_a)
    oscprob_ma.oscprob.Enu(E)
    data_osc_ma = oscprob_ma.oscprob.oscprob

    #Vacuum neutrino (same antineutrino)
    oscprob = ROOT.OscProbPMNS(from_nu, to_nu)
    oscprob.full_osc_prob_gpu.inputs.Enu(E)
    data_osc = oscprob.full_osc_prob_gpu.oscprob




plt.plot(E_arr*1e-3, data_osc.data())
plt.plot(E_arr*1e-3, data_osc_m.data())
plt.plot(E_arr*1e-3, data_osc_ma.data())
plt.legend(('Vacuum', 'Matter nu', 'Matter a_nu'))
plt.xlabel('$E, GeV $')
plt.ylabel(r'$P_{\nu_{\mu} \to \nu_{e}}$')
plt.grid()
plt.show()
~          
