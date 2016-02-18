# coding: utf-8

import ROOT
import numpy as np
from collections import defaultdict
from gna import datapath
import gna.parameters.ibd
import gna.parameters.oscillation
import itertools
from gna.exp import baseexp

class Reactor(object):
    def __init__(self, ns, name, location, fission_fractions,
                 power=None, power_rate=None):
        self.ns = ns("reactors")(name)
        self.name = name
        self.location = location

        if power is not None:
            self.power = power
            self.ns.defparameter("ThermalPower", central=power, sigma=0)
        else:
            self.power = None

        if power_rate is not None:
            self.power_rate = ROOT.Points(power_rate)
        else:
            self.power_rate = None

        self.fission_fractions = {isoname: ROOT.Points(frac)
                                  for isoname, frac in fission_fractions.iteritems()}

    def calcdistance(self, detector):
        return np.sqrt(np.sum((self.location - detector.location)**2))

    def initdistance(self, ns, detector):
        dist = self.calcdistance(detector)
        pair_ns = ns("pairs")(self.name)(detector.name)
        pair_ns.defparameter("L", central=dist, sigma=0)

class ReactorGroup(object):
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

class Detector(object):
    def __init__(self, ns, name, location,
                 protons=None, livetime=None):
        self.ns = ns("detectors")(name)
        self.name = name
        self.location = location

        if protons is not None:
            self.protons = protons
            self.ns.defparameter("TargetProtons", central=protons, sigma=0)
        else:
            self.protons = None

        if livetime is not None:
            self.livetime = ROOT.Points(livetime)
        else:
            self.livetime = None

        self.components = defaultdict(ROOT.Sum)
        self.hists = dict()

        self.ns("eres").defparameter("Eres_a", central=0.0, sigma=0)
        self.ns("eres").defparameter("Eres_b", central=0.03, sigma=0)
        self.ns("eres").defparameter("Eres_c", central=0.0, sigma=0)

class Isotope(object):
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

        self.Es, self.ys = np.loadtxt(datapath(self.spectrumfiles[name]), unpack=True)
        self.spectrum = ROOT.LinearInterpolator(len(self.Es), self.Es.copy(), self.ys.copy())

        self.eperfission = self.eperfission[name]

class LazyConnector(object):
    def __init__(self):
        self.inputs = defaultdict(set)
        self.allinputs = set()
        self.outputs = {}

    def _connect(self):
        for inpname in self.inputs:
            out = self.outputs.get(inpname, None)
            if out:
                for inp in self.inputs[inpname]:
                    inp.connect(out)
                self.inputs[inpname].clear()

    def add(self, name, io):
        if isinstance(io, ROOT.OutputDescriptor):
            self.outputs[name] = io
        elif isinstance(io, ROOT.InputDescriptor):
            if io in self.allinputs:
                return
            self.allinputs.add(io)
            self.inputs[name].add(io)
        self._connect()

class ReactorExperimentModel(baseexp):
    @classmethod
    def initparser(self, parser, env):
        def OscProb(name):
            return {
                'standard': ROOT.OscProbPMNS,
                'decoh': ROOT.OscProbPMNSDecoh,
            }[name]
        parser.add_argument('--ns')
        parser.add_argument('--ibd', choices=['zero', 'first'], default='zero')
        parser.add_argument('--oscprob', choices=['standard', 'decoh'],
                            type=OscProb, default='standard')
        parser.add_argument('--erange', type=float, nargs=2,
                            default=[1.0, 8.0],
                            metavar=('E_MIN', 'E_MAX'),
                            help='energy range')
        parser.add_argument('--nbins', type=int,
                            default=200,
                            help='number of bins')
        parser.add_argument('--integration-order', type=int, default=10)

    def __init__(self, opts, name=None, ns=None, reactors=None, detectors=None):
        super(ReactorExperimentModel, self).__init__(opts)
        self._oscprobs = {}
        self.connector = LazyConnector()

        self.name = name

        self.ns = ns or env.ns(opts.ns or self.name)
        self.reqparameters(self.ns)

        self.reactors = self.makereactors(reactors)
        self.detectors = self.makedetectors(detectors)

        for reactor in self.reactors:
            reactor.isotopes = [Isotope(self.ns, isoname) for isoname in reactor.fission_fractions]

        self.linkpairs(self.reactors, self.detectors)

        for reactor in self.reactors:
            for isotope in reactor.isotopes:
                self.connector.add('Enu', isotope.spectrum.f.inputs.x)

        edges = np.linspace(opts.erange[0], opts.erange[1], opts.nbins+1)
        orders = np.array([10]*(len(edges)-1), dtype=int)
        self.setibd(edges, orders, opts.ibd)

        for detector in self.detectors:
            self.setupobservations(detector)

    def _getoscprobcls(self, reactor, detector):
        """Returns oscillation probability class for given (reactor,detector)"""
        if isinstance(reactor, ReactorGroup):
            return (ROOT.OscProbPMNSMult, ROOT.OscProbPMNS)
        keys = [frozenset([reactor, detector]), reactor, detector, None]
        for key in keys:
            try:
                return (self._oscprobs[key], self._oscprobs[key])
            except KeyError:
                pass
        return (ROOT.OscProbPMNS, ROOT.OscProbPMNS)

    def _getnormtype(self, reactor, detector):
        if all(x is not None for x in [reactor.power, reactor.power_rate, detector.protons, detector.livetime]):
            return 'calc'
        return 'manual'

    def groupreactors(self, reactors, detector, precision=1e-2):
        """Merges some Reactors of reactors to ReactorGroup if they are close enough

        Oscillation probability error order is precision**3
        """
        groups = []
        groupable = []
        for reactor in reactors:
            if self._getoscprobcls(reactor, detector)[0] is ROOT.OscProbPMNS and self._getnormtype(reactor, detector) == 'calc':
                groupable.append(reactor)
            else:
                groups.append([reactor])
        if not groupable:
            return reactors
        groups.append([])
        for reactor in sorted(groupable, key=lambda r: r.calcdistance(detector)):
            groups[-1].append(reactor)
            L = np.array([r.calcdistance(detector) for r in groups[-1]])
            P = np.array([r.power for r in groups[-1]])
            Lavg = sum(P/L)/sum(P/L**2)
            if not all(abs(L/Lavg-1) < precision):
                groups.append([groups[-1].pop()])
        grouped = []
        cnt = 0
        for group in groups:
            if len(group) > 1:
                rgroup = ReactorGroup(self.ns, "group{}".format(cnt), group)
                rgroup.initdistance(self.ns, detector)
                grouped.append(rgroup)
                cnt += 1
            else:
                grouped.append(group[0])
        return grouped

    def normalization(self, reactor, detector, normtype):
        """Returns normalization object for given reactor/detector pair"""
        def vec(lst):
            v = ROOT.vector("std::string")()
            for s in lst:
                v.push_back(s)
            return v

        if normtype == 'calc':
            bindings = {}
            for isotope in reactor.isotopes:
                bindings["EnergyPerFission_{0}".format(isotope.name)] = self.ns("isotopes")(isotope.name)["EnergyPerFission"]
            norm = ROOT.ReactorNorm(vec(iso.name for iso in reactor.isotopes), bindings=bindings)
            norm.isotopes.livetime(detector.livetime)
            norm.isotopes.power_rate(reactor.power_rate)
            for isoname, frac in reactor.fission_fractions.iteritems():
                norm.isotopes['fission_fraction_{0}'.format(isoname)](frac)
        elif normtype == 'manual':
            norm = ROOT.ReactorNormAbsolute(vec(iso.name for iso in reactor.isotopes))
            for isoname, frac in reactor.fission_fractions.iteritems():
                norm.isotopes['fission_fraction_{0}'.format(isoname)](frac)
        return norm

    def linkpairs(self, reactors, detectors):
        for reactor, detector in itertools.product(reactors, detectors):
            reactor.initdistance(self.ns, detector)

        for detector in detectors:
            grouped = self.groupreactors(reactors, detector)
            for rgroup in grouped:
                pair_ns = self.ns("pairs")(rgroup.name)(detector.name)
                normtype = self._getnormtype(rgroup, detector)
                with detector.ns, rgroup.ns, pair_ns:
                    norm = self.normalization(rgroup, detector, normtype)
                    oscprobcls, weightscls = self._getoscprobcls(rgroup, detector)
                    with self.ns("oscillation"):
                        oscprob = oscprobcls(ROOT.Neutrino.ae(), ROOT.Neutrino.ae())

                normedflux = ROOT.Sum()
                for isotope in rgroup.isotopes:
                    subflux = ROOT.Product()
                    subflux.multiply(isotope.spectrum)
                    subflux.multiply(norm.isotopes['norm_{0}'.format(isotope.name)])
                    normedflux.add(subflux)
                compnames = set(oscprob.probsum.inputs)
                detector.components[(weightscls, 'comp0')].add(normedflux)
                compnames.remove('comp0')
                for osccomps in oscprob.transformations.itervalues():
                    for compname, osccomp in osccomps.outputs.iteritems():
                        if compname not in compnames:
                            continue
                        product = ROOT.Product()
                        product.multiply(normedflux)
                        product.multiply(osccomp)
                        detector.components[(weightscls, compname)].add(product)
                        if compname not in compnames:
                            raise Exception("overriden component {}".format(compname))
                        compnames.remove(compname)
                        if 'Enu' in osccomps.inputs:
                            self.connector.add('Enu', osccomps.inputs.Enu)
                if compnames:
                    raise Exception("components not found: {}".format(compnames))

    def setibd(self, edges, orders, ibdtype):
        if ibdtype == 'zero':
            with self.ns("ibd"):
                ibd = ROOT.IbdZeroOrder()
            integrator = ROOT.GaussLegendre(edges, orders, len(orders))
            histcls = ROOT.GaussLegendreHist
            ibd.xsec.inputs(integrator.points)
            ibd.Enu.inputs(integrator.points)
            eventsparts = [ibd.xsec]
        elif ibdtype == 'first':
            with self.ns("ibd"):
                ibd = ROOT.IbdFirstOrder()
            integrator = ROOT.GaussLegendre2d(edges, orders, len(orders), -1.0, 1.0, 5)
            histcls = ROOT.GaussLegendre2dHist
            ibd.Enu.inputs(integrator.points)
            ibd.xsec.Enu(ibd.Enu)
            ibd.xsec.ctheta(integrator.points.y)
            ibd.jacobian.Enu(ibd.Enu)
            ibd.jacobian.Ee(integrator.points.x)
            ibd.jacobian.ctheta(integrator.points.y)
            eventsparts = [ibd.xsec, ibd.jacobian]
        else:
            raise Exception("unknown ibd type {0!r}".format(ibdtype))

        self.connector.add('Enu', ibd.Enu.Enu)
        for detector in self.detectors:
            for compid, comp in detector.components.iteritems():
                product = ROOT.Product()
                for part in eventsparts:
                    product.multiply(part)
                product.multiply(comp)
                detector.hists[compid] = histcls(integrator)
                detector.hists[compid].hist.inputs(product)

    def setupobservations(self, detector):
        oscprobs = {}
        for oscprobcls, compname in detector.components:
            if oscprobcls in oscprobs:
                continue
            with self.ns("oscillation"):
                oscprob = oscprobcls(ROOT.Neutrino.ae(), ROOT.Neutrino.ae(),
                                     freevars=['L'])
            oscprobs[oscprobcls] = oscprob

        probsums = []
        for compid, hist in detector.hists.iteritems():
            oscprob = oscprobs[compid[0]]
            oscprob.probsum[compid[1]](hist.hist)
            self.ns.addobservable("{0}_{1}".format(detector.name, compid[1]), hist.hist)
            probsums.append(oscprob.probsum)
        if len(probsums) > 1:
            finalsum = ROOT.Sum()
            for probsum in probsums:
                finalsum.add(probsum)
        else:
            finalsum = probsums[0]
        self.ns.addobservable("{0}_noeffects".format(detector.name), finalsum)
        with detector.ns("eres"):
            detector.eres = ROOT.EnergyResolution()
        detector.eres.smear.inputs(finalsum)
        self.ns.addobservable("{0}".format(detector.name), finalsum)

    @classmethod
    def reqparameters(cls, ns):
        gna.parameters.ibd.reqparameters(ns("ibd"))
        gna.parameters.oscillation.reqparameters(ns("oscillation"))

        for isoname, (central, sigma) in Isotope.eperfission.iteritems():
            isons = ns("isotopes")(isoname)
            isons.reqparameter("EnergyPerFission", central=central, sigma=sigma)

    @classmethod
    def makeisotopes(cls, ns):
        return [Isotope(ns, isoname) for isoname in Isotope.spectrumfiles]

    def makereactors(self, reactors):
        return reactors

    def makedetectors(self, detectors):
        return detectors
