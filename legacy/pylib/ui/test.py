from gna.ui import basecmd
import ROOT
import numpy as np
import physlib
from gna import data
from matplotlib import pyplot as plt

def loadspectrum(fname):
    Es, ys = data.load(fname).T
    return ROOT.LinearInterpolator(len(Es), Es.copy(), ys.copy())

def vec(lst):
    v = ROOT.vector("std::string")()
    for s in lst:
        v.push_back(s)
    return v

spectra = {
    'Pu239': ('Huber_smooth_extrap_Pu239_13MeV0.01MeVbin.dat',  0.60),
    'Pu241': ('Huber_smooth_extrap_Pu241_13MeV0.01MeVbin.dat',  0.07),
    'U235':  ('Huber_smooth_extrap_U235_13MeV0.01MeVbin.dat',   0.27),
    'U238':  ('Mueller_smooth_extrap_U238_13MeV0.01MeVbin.dat', 0.06),
}
isotopes = spectra.keys()

class cmd(basecmd):
    def setupreactor(self):
        env.defparameter('NominalThermalPower', central=2.0, sigma=0.01)

        for isoname in spectra.items():
            env.defparameter("weight_{0}".format(isoname), central=weight, sigma=0)

    def run(self):
        env = self.env.default
        env.defparameter('SinSq12', central=0.307, sigma=0.017, limits=(0, 1))
        env.defparameter('SinSq13', central=0.0214, sigma=0.002, limits=(0, 1))
        env.defparameter('SinSq23', central=0.42, sigma=0.08, limits=(0, 1))

        env.defparameter('DeltaMSq12', central=7.54e-5, sigma=0.24e-5,
                         limits=(0, 0.1))
        env.defparameter('DeltaMSqEE', central=2.44e-3, sigma=0.10e-3,
                         limits=(0, 0.1))
        env.defparameter('Alpha', type='discrete', default='normal',
                         variants={'normal': 1.0, 'inverted': -1.0})

        env.defparameter('Delta', type='uniformangle', central=0.0)

        env.defparameter('L', central=52, sigma=0)

        pdg = physlib.pdg[2012]
        env.defparameter("NeutronLifeTime", central=pdg['neutron_lifetime'],
                         sigma=0)
        env.defparameter("ProtonMass", central=pdg['ProtonMass'], sigma=0)
        env.defparameter("NeutronMass", central=pdg['NeutronMass'], sigma=0)
        env.defparameter("ElectronMass", central=pdg['ElectronMass'], sigma=0)

        with env.ns("eres"):
            env.defparameter("a", central=0.0, sigma=0)
            env.defparameter("b", central=0.03, sigma=0)
            env.defparameter("c", central=0.0, sigma=0)

        expressions = []
        expressions.append(ROOT.OscillationExpressions())
        expressions.append(ROOT.PMNSExpressions())

        edges = np.linspace(0, 10, 500)
        orders = np.array([5]*(len(edges)-1), dtype=int)

        prediction = ROOT.Concat()
        integrator = ROOT.GaussLegendre2d(edges, orders, len(orders), -1.0, 1.0, 5)
        ibd = ROOT.IbdFirstOrder()
        ibd.Enu.inputs(integrator.points.x)
        ibd.xsec.Enu(ibd.Enu)
        ibd.xsec.ctheta(integrator.points.y)
        ibd.jacobian.Enu(ibd.Enu)
        ibd.jacobian.Ee(integrator.points.x)
        ibd.jacobian.ctheta(integrator.points.y)
        spectrum = ROOT.WeightedSum(vec([a+'_weight' for a in spectra.keys()]), vec(spectra.keys()))
        for isoname, (fname, weight) in spectra.items():
            isotope = loadspectrum(fname)
            isotope.f.inputs(ibd.Enu)
            spectrum.sum[isoname](isotope.f)
        oscprob = ROOT.OscProbPMNS(ROOT.Neutrino.ae(), ROOT.Neutrino.ae())
        components = {}
        for compname in oscprob.transformations:
            if not compname.startswith('comp'):
                continue
            events = ROOT.Product()
            events.multiply(spectrum)
            events.multiply(ibd.xsec)
            oscprob[compname].inputs(ibd.Enu)
            events.multiply(oscprob[compname])
            oscprob.probsum.inputs[compname](events)
            components[compname] = events.product
        if 'comp0' in oscprob.probsum.inputs:
            events = ROOT.Product()
            events.multiply(spectrum)
            events.multiply(ibd.xsec)
            oscprob.probsum.comp0(events)
        integrator.hist.inputs(oscprob.probsum)
        eres = ROOT.EnergyResolutionC(ns="eres")
        eres.smear.inputs(integrator.hist)
        prediction.append(eres.smear)
        import time
        t = time.time()
        ibd0 = 1e25*np.frombuffer(prediction.data(), dtype=float, count=prediction.size()).copy()
        print time.time() - t
        env.pars["DeltaMSqEE"].set(2e-3)
        t = time.time()
        ibd0 = 1e25*np.frombuffer(prediction.data(), dtype=float, count=prediction.size()).copy()
        print time.time() - t
        plt.plot((edges[:-1] + edges[1:])/2, ibd0)
        plt.show()
        # print ibd0

        return
        # prediction = ROOT.Concat()
        # integrator = ROOT.GaussLegendre2d(edges, orders, len(orders), -1.0, 1.0, 5)
        # events = ROOT.Product()
        # events.multiply(ibd.jacobian)
        # oscprob = ROOT.OscProb2nu()
        # oscprob.prob.inputs(ibd.Enu)
        # events.multiply(oscprob.prob)
        # integrator.hist.inputs(events.product)
        # prediction.append(integrator.hist)
        # ibd1 = np.frombuffer(prediction.data(), dtype=float, count=prediction.size()).copy()
        # print ibd1

        print ibd1/ibd0
