import ROOT
import numpy as np
from collections import defaultdict
from gna import datapath
import gna.parameters.ibd
import gna.parameters.oscillation

class reactor(object):
    def __init__(self, ns, name, location, power, power_rate,
                 isotopes, fission_fractions):
        self.ns = ns("reactors")(name)
        self.name = name
        self.location = location

        self.power = power
        self.ns.defparameter("ThermalPower", central=power, sigma=0)

        self.power_rate = ROOT.Points(power_rate)

        self.isotopes = isotopes
        self.fission_fractions = {isoname: ROOT.Points(frac)
                                  for isoname, frac in fission_fractions.iteritems()}

    def calcdistance(self, detector):
        return np.sqrt(np.sum((self.location - detector.location)**2))

    def initdistance(self, ns, detector):
        dist = self.calcdistance(detector)
        pair_ns = ns("pairs")(self.name)(detector.name)
        pair_ns.defparameter("L", central=dist, sigma=0)

    def makeoscprob(self):
        return ROOT.OscProbPMNS(ROOT.Neutrino.ae(), ROOT.Neutrino.ae())

def makereactors(ns, common, data):
    return [reactor(ns, **dict(common.items(), **x)) for x in data]

class reactorgroup(object):
    def __init__(self, ns, name, reactors):
        self.ns = ns("reactorgroups")(name)
        self.name = name
        self.reactors = reactors

        self.power_rate = reactors[0].power_rate

        self.isotopes = reactors[0].isotopes
        self.fission_fractions = reactors[0].fission_fractions

    def initdistance(self, ns, detector):
        pair_ns = ns("pairs")(self.name)(detector.name)
        bindings = {}
        for i, reactor in enumerate(self.reactors):
            bindings["L_{0}".format(i)] = ns("pairs")(reactor.name)(detector.name)["L"]
            bindings["P_{0}".format(i)] = reactor.ns["ThermalPower"]
        ROOT.ReactorGroup(len(self.reactors), ns=pair_ns, bindings=bindings)

        pair_ns.defparameter("L", target=pair_ns.ref("Lavg"))
        pair_ns.defparameter("ThermalPower", target=pair_ns.ref("Pavg"))

    def makeoscprob(self):
        return ROOT.OscProbPMNSMult(ROOT.Neutrino.ae(), ROOT.Neutrino.ae())

class detector(object):
    def __init__(self, ns, name, location, protons, livetime):
        self.ns = ns("detectors")(name)
        self.name = name
        self.location = location

        self.protons = protons
        self.ns.defparameter("TargetProtons", central=protons, sigma=0)

        self.livetime = ROOT.Points(livetime)

        self.components = defaultdict(ROOT.Sum)

        self.ns("eres").defparameter("a", central=0.0, sigma=0)
        self.ns("eres").defparameter("b", central=0.03, sigma=0)
        self.ns("eres").defparameter("c", central=0.0, sigma=0)

def makedetectors(ns, common, data):
    return [detector(ns, **dict(common.items(), **x)) for x in data]

class isotope(object):
    spectrumfiles = {
        'Pu239': 'Huber_smooth_extrap_Pu239_13MeV0.01MeVbin.dat',
        'Pu241': 'Huber_smooth_extrap_Pu241_13MeV0.01MeVbin.dat',
        'U235': 'Huber_smooth_extrap_U235_13MeV0.01MeVbin.dat',
        'U238': 'Mueller_smooth_extrap_U238_13MeV0.01MeVbin.dat',
    }
    eperfission = {
        'Pu239': (209.99, 0.60),
        'Pu241': (213.60, 0.65),
        'U235': (201.92, 0.46),
        'U238': (205.52, 0.96),
    }

    def __init__(self, ns, name):
        self.ns = ns("isotopes")(name)
        self.name = name

        self.Es, self.ys = np.loadtxt(datapath(isotope.spectrumfiles[name]), unpack=True)
        self.spectrum = ROOT.LinearInterpolator(len(self.Es), self.Es.copy(), self.ys.copy())

        self.eperfission = isotope.eperfission[name]

def makeisotopes(ns):
    return [isotope(ns, isoname) for isoname in isotope.spectrumfiles]

def makenorm(ns, reactor, detector):
    def vec(lst):
        v = ROOT.vector("std::string")()
        for s in lst:
            v.push_back(s)
        return v

    bindings = {}
    for isotope in reactor.isotopes:
        bindings["EnergyPerFission_{0}".format(isotope.name)] = ns("isotopes")(isotope.name)["EnergyPerFission"]
    norm = ROOT.ReactorNorm(vec(iso.name for iso in reactor.isotopes), bindings=bindings)
    norm.isotopes.livetime(detector.livetime)
    norm.isotopes.power_rate(reactor.power_rate)
    for isoname, frac in reactor.fission_fractions.iteritems():
        norm.isotopes['fission_fraction_{0}'.format(isoname)](frac)
    return norm

def oscflux(ns, react, detector):
    with ns("oscillation"), detector.ns, react.ns, ns("pairs")(react.name)(detector.name):
        norm = makenorm(ns, react, detector)
        oscprob = react.makeoscprob()
    return norm, oscprob

def makecomponents(isotopes, norm, oscprob):
    normedflux = ROOT.Sum()
    for isotope in isotopes:
        subflux = ROOT.Product()
        subflux.multiply(isotope.spectrum)
        subflux.multiply(norm.isotopes['norm_{0}'.format(isotope.name)])
        normedflux.add(subflux)
    components = {
        'comp0': normedflux,
    }
    for compname in (x for x in oscprob.transformations if x.startswith('comp')):
        oscflux = ROOT.Product()
        oscflux.multiply(normedflux)
        oscflux.multiply(oscprob[compname])
        components[compname] = oscflux
    return components

def setupcomponents(ns, reactors, detectors, Enu):
    for detector in detectors:
        for reactor in reactors:
            reactor.initdistance(ns, detector)

    for detector in detectors:
        groups = [[]]
        for reactor in sorted(reactors, key=lambda r: r.calcdistance(detector)):
            groups[-1].append(reactor)
            L = np.array([r.calcdistance(detector) for r in groups[-1]])
            P = np.array([r.power for r in groups[-1]])
            Lavg = sum(P/L)/sum(P/L**2)
            if not all(abs(L/Lavg-1) < 1e-2):
                groups.append([groups[-1].pop()])
        for rgroup in groups:
            if len(rgroup) == 1:
                react = rgroup[0]
            else:
                react = reactorgroup(ns, '_'.join(r.name for r in rgroup), rgroup)
                react.initdistance(ns, detector)
            norm, oscprob = oscflux(ns, react, detector)
            for compname in (x for x in oscprob.transformations if x.startswith('comp')):
                oscprob[compname].inputs(Enu)
            for compname, comp in makecomponents(rgroup[0].isotopes, norm, oscprob).iteritems():
                detector.components[compname].add(comp)

class component(object):
    def __init__(self, integrator, eventsparts):
        self.integrator = integrator
        events = ROOT.Product()
        for part in eventsparts:
            events.multiply(part)
        idx = len(integrator.hist.inputs)
        integrator.addfunction(events)
        self.hist = integrator.hist.outputs[idx]

def init(ns, edges, orders, ibdtype):
    eventsparts = []
    if ibdtype == 'zero':
        with ns("ibd"):
            ibd = ROOT.IbdZeroOrder()
        integrator = ROOT.GaussLegendre(edges, orders, len(orders))
        ibd.xsec.inputs(integrator.points)
        ibd.Enu.inputs(integrator.points)
        eventsparts.append(ibd.xsec)
    elif ibdtype == 'first':
        with ns("ibd"):
            ibd = ROOT.IbdFirstOrder()
        integrator = ROOT.GaussLegendre2d(edges, orders, len(orders), -1.0, 1.0, 5)
        ibd.Enu.inputs(integrator.points)
        ibd.xsec.Enu(ibd.Enu)
        ibd.xsec.ctheta(integrator.points.y)
        ibd.jacobian.Enu(ibd.Enu)
        ibd.jacobian.Ee(integrator.points.x)
        ibd.jacobian.ctheta(integrator.points.y)
        eventsparts.extend([ibd.xsec, ibd.jacobian])
    else:
        raise Exception("unknown ibd type {0!r}".format(ibdtype))

    def integratecomp(compname, inp):
        return component(integrator, eventsparts+[inp])

    return ibd.Enu, integratecomp

def setupobservations(ns, detector, compfactory):
    with ns("oscillation"):
        detector.oscprob = ROOT.OscProbPMNS(ROOT.Neutrino.ae(), ROOT.Neutrino.ae(),
                                            freevars=['L'])
    integrated = {}
    for compname in detector.components:
        integrated[compname] = compfactory(compname, detector.components[compname])
        detector.oscprob.probsum[compname](integrated[compname].hist)
    detector.integrated = integrated
    ns.addobservable("{0}_noeffects".format(detector.name), detector.oscprob.probsum)
    with detector.ns("eres"):
        detector.eres = ROOT.EnergyResolution()
    detector.eres.smear.inputs(detector.oscprob.probsum)
    ns.addobservable("{0}".format(detector.name), detector.eres.smear)

def defparameters(ns):
    gna.parameters.ibd.defparameters(ns("ibd"))
    gna.parameters.oscillation.defparameters(ns("oscillation"))
    for isoname, (central, sigma) in isotope.eperfission.iteritems():
        isons = ns("isotopes")(isoname)
        isons.defparameter("EnergyPerFission", central=central, sigma=sigma)
