from gna.exp import baseexp
from gna import data
import ROOT
import numpy as np
from collections import defaultdict
from itertools import chain

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

eperfission = {
    'U235'  : [ 201.92, 0.46 ],
    'U238'  : [ 205.52, 0.96 ],
    'Pu239' : [ 209.99, 0.60 ],
    'Pu241' : [ 213.60, 0.65 ]
}

class reactor(object):
    def __init__(self, ns, name, power, location):
        self.ns = ns(name)
        self.name = name
        self.location = location

        self.power = power
        self.ns.defparameter("ThermalPower", central=power, sigma=0)

        self.power_rate = ROOT.Points([1.0])

        self.fission_fractions = {isoname: ROOT.Points([frac])
                                  for isoname, (fname, frac) in spectra.iteritems()}

    def calcdistance(self, detector):
        return np.sqrt(np.sum((self.location - detector.location)**2))

class detector(object):
    def __init__(self, ns, name, protons, location):
        self.ns = ns(name)
        self.name = name
        self.location = location

        self.protons = protons
        self.ns.defparameter("TargetProtons", central=protons, sigma=0)

        self.livetime = ROOT.Points([6*365*24*60*60])

        self.components = defaultdict(ROOT.Sum)

        self.ns.defparameter("a", central=0.0, sigma=0)
        self.ns.defparameter("b", central=0.03, sigma=0)
        self.ns.defparameter("c", central=0.0, sigma=0)

class exp(baseexp):
    @classmethod
    def initparser(self, parser):
        parser.add_argument('--erange', type=float, nargs=2,
                            default=[1.0, 10.0],
                            metavar=('E_MIN', 'E_MAX'),
                            help='energy range')
        parser.add_argument('--nbins', type=int,
                            default=200,
                            help='number of bins')

    def makespectra(self):
        res = {}
        for isoname, (fname, frac) in spectra.iteritems():
            Es, ys = data.load(fname).T
            res[isoname] = ROOT.LinearInterpolator(len(Es), Es.copy(), ys.copy())
        return res

    def distance(self, reactor, detector):
        ns = self.ns('.'.join(["distances", reactor.name, detector.name]))
        if 'L' not in ns:
            ns.defparameter("L", central=reactor.calcdistance(detector), sigma=0)
        return ns.getparameter('L')

    def makenorm(self, reactor, detector):
        with detector.ns:
            norm = ROOT.ReactorNorm(vec(isotopes))
        norm.isotopes.livetime(detector.livetime)
        norm.isotopes.power_rate(reactor.power_rate)
        for isoname, frac in reactor.fission_fractions.iteritems():
            norm.isotopes['fission_fraction_{0}'.format(isoname)](frac)
        return norm

    def initreactor(self, reactor, detector):
        with self.env.bind(L=self.distance(reactor, detector),
                           ThermalPower=reactor.ns["ThermalPower"]):
            oscprob = ROOT.OscProbPMNS(ROOT.Neutrino.ae(), ROOT.Neutrino.ae())
            return self.makenorm(reactor, detector), oscprob

    def initreactorgroup(self, reactors, detector):
        group_ns = self.ns(["reactorgroups",
                            ",".join([r.name for r in reactors]),
                            detector.name])
        bindings = {}
        for i, reactor in enumerate(reactors):
            bindings["L_{0}".format(i)] = self.distance(reactor, detector)
            bindings["P_{0}".format(i)] = reactor.ns["ThermalPower"]
        with self.env.bind(**bindings):
            ROOT.ReactorGroup(len(reactors), ns=group_ns)
        with group_ns, self.env.bind(L='Lavg', ThermalPower='Pavg'):
            oscprob = ROOT.OscProbPMNSMult(ROOT.Neutrino.ae(), ROOT.Neutrino.ae())
            return self.makenorm(reactors[0], detector), oscprob

    def getcomponents(self, norm, oscprob):
        normedflux = ROOT.Sum()
        for isoname, spectrum in self.isospectra.iteritems():
            subflux = ROOT.Product()
            subflux.multiply(spectrum)
            subflux.multiply(norm.isotopes['norm_{0}'.format(isoname)])
            normedflux.add(subflux)
        components = {
            'comp0': normedflux,
        }
        for compname in oscprob.transformations:
            if not compname.startswith('comp'):
                continue
            oscflux = ROOT.Product()
            oscflux.multiply(normedflux)
            oscflux.multiply(oscprob[compname])
            components[compname] = oscflux
        return components

    def makereactors(self):
        ns = self.env.ns("reactors")
        reactors = []

        reactors.append(reactor(ns, name='YJ1', location=52.75, power=2.9))
        reactors.append(reactor(ns, name='YJ2', location=52.84, power=2.9))
        reactors.append(reactor(ns, name='YJ3', location=52.42, power=2.9))
        reactors.append(reactor(ns, name='YJ4', location=52.51, power=2.9))
        reactors.append(reactor(ns, name='YJ5', location=52.12, power=2.9))
        reactors.append(reactor(ns, name='YJ6', location=52.21, power=2.9))

        reactors.append(reactor(ns, name='TS1', location=52.76, power=4.6))
        reactors.append(reactor(ns, name='TS2', location=52.63, power=4.6))
        reactors.append(reactor(ns, name='TS3', location=52.32, power=4.6))
        reactors.append(reactor(ns, name='TS4', location=52.20, power=4.6))

        reactors.append(reactor(ns, name='DYB', location=215.0, power=17.4))
        reactors.append(reactor(ns, name='HZ', location=265.0, power=17.4))

        return reactors

    def makedetectors(self):
        ns = self.env.ns("detectors")
        detectors = []
        detectors.append(detector(ns, name='AD1', location=0.0, protons=1.42e33))
        return detectors

    def init(self):
        self.ns = self.env.ns('juno')

        for isoname, e in eperfission.iteritems():
            self.env.defparameter("EnergyPerFission_{0}".format(isoname),
                                  central=e[0], sigma=e[1])

        edges = np.linspace(self.opts.erange[0], self.opts.erange[1], self.opts.nbins+1)
        orders = np.array([10]*(len(edges)-1), dtype=int)

        self.isospectra = self.makespectra()
        self.reactors = self.makereactors()
        self.detectors = self.makedetectors()

        ibd = ROOT.IbdFirstOrder()

        for detector in self.detectors:
            groups = [[]]
            for reactor in sorted(self.reactors, key=lambda r: r.calcdistance(detector)):
                groups[-1].append(reactor)
                L = np.array([r.calcdistance(detector) for r in groups[-1]])
                P = np.array([r.power for r in groups[-1]])
                Lavg = sum(P/L)/sum(P/L**2)
                if not all(abs(L/Lavg-1) < 1e-2):
                    groups.append([groups[-1].pop()])
            for reactors in groups:
                if len(reactors) == 1:
                    norm, oscprob = self.initreactor(reactors[0], detector)
                else:
                    norm, oscprob = self.initreactorgroup(reactors, detector)
                for compname in oscprob.transformations:
                    if compname.startswith('comp'):
                        oscprob[compname].inputs(ibd.Enu)
                for compname, comp in self.getcomponents(norm, oscprob).iteritems():
                    detector.components[compname].add(comp)

        for detector in self.detectors:
            detector.oscprob = ROOT.OscProbPMNS(ROOT.Neutrino.ae(), ROOT.Neutrino.ae(),
                                                freevars=['L'])
            integrators = {}
            for compname in detector.components:
                integrator = ROOT.GaussLegendre2d(edges, orders, len(orders), -1.0, 1.0, 5)
                integrators[compname] = integrator
                compevents = ROOT.Product()
                compevents.multiply(detector.components[compname])
                compevents.multiply(ibd.xsec)
                compevents.multiply(ibd.jacobian)
                integrators[compname].hist.inputs(compevents)

                detector.components[compname] = integrators[compname]
                detector.oscprob.probsum[compname](detector.components[compname].hist)

        ibd.Enu.inputs(integrator.points)
        for isoname, spectrum in self.isospectra.iteritems():
            spectrum.f.inputs(ibd.Enu)
        ibd.xsec.Enu(ibd.Enu)
        ibd.xsec.ctheta(integrator.points.y)
        ibd.jacobian.Enu(ibd.Enu)
        ibd.jacobian.Ee(integrator.points.x)
        ibd.jacobian.ctheta(integrator.points.y)

        self.ns.addobservable("spectrum", self.detectors[0].oscprob.probsum)
