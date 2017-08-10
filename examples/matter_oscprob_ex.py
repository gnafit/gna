import numpy as np
import matplotlib.pyplot as plt
import ROOT
from gna.ui import basecmd
import gna.parameters.oscillation

self.ns = self.env.ns("004")
gna.parameters.oscillation.reqparameters(self.ns)
self.ns.defparameter("L", central=810,sigma=0) #километры
self.ns.defparameter("rho",central=2.7,sigma=0) #г/см3

from_nu_a = ROOT.Neutrino.amu()
to_nu_a = ROOT.Neutrino.ae()

from_nu = ROOT.Neutrino.mu()
to_nu = ROOT.Neutrino.e()

E_arr = np.array(range(500, 6000, 50))  #энергия в МеV

E = ROOT.Points(E_arr)


with self.ns:
    #нейтрино в материи
    oscprob_m = ROOT.OscProbMatter(from_nu, to_nu)
    oscprob_m.oscprob.Enu(E)
    data_osc_m = oscprob_m.oscprob.oscprob

    #антинейтрино в материи
    oscprob_ma = ROOT.OscProbMatter(from_nu_a, to_nu_a)
    oscprob_ma.oscprob.Enu(E)
    data_osc_ma = oscprob_ma.oscprob.oscprob

    #нейтрино в вакууме (аналогично антинейтрино в вакууме)
    oscprob = ROOT.OscProbPMNS(from_nu, to_nu)
    oscprob.full_osc_prob.inputs.Enu(E)
    data_osc = oscprob.full_osc_prob.oscprob




plt.plot(E_arr*1e-3, data_osc.data())
plt.plot(E_arr*1e-3, data_osc_m.data())
plt.plot(E_arr*1e-3, data_osc_ma.data())
plt.legend(('Vacuum', 'Matter nu', 'Matter a_nu'))
plt.xlabel('$E, GeV $')
plt.ylabel(r'$P_{\nu_{\mu} \to \nu_{e}}$')
plt.grid()
plt.show()
