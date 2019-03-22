# coding: utf-8

import ROOT
from scipy import integrate as integrate
from gna import constructors as C
from ROOT import TFile, TProfile, TGraphAsymmErrors
from ROOT import gROOT
import numpy as np
from collections import defaultdict
from gna import datapath
import gna.parameters.ibd
import gna.parameters.oscillation
import itertools
from gna.exp import baseexp
from gna.env import env


def vec(lst):
    v = ROOT.vector("std::string")()
    for s in lst:
        v.push_back(str(s))
    return v

year=365*24*60*60.0

spectrumfiles = {
    'Pu239': 'reactor_anu_spectra/Huber/Huber_smooth_extrap_Pu239_13MeV0.01MeVbin.dat',
    'Pu241': 'reactor_anu_spectra/Huber/Huber_smooth_extrap_Pu241_13MeV0.01MeVbin.dat',
    'U235': 'reactor_anu_spectra/Huber/Huber_smooth_extrap_U235_13MeV0.01MeVbin.dat',
    'U238': 'reactor_anu_spectra/Mueller/Mueller_smooth_extrap_U238_13MeV0.01MeVbin.dat',
}

eperfission = {
    'Pu239': (209.99, 0.60),
    'Pu241': (213.60, 0.65),
    'U235': (201.92, 0.46),
    'U238': (205.52, 0.96),
}

geo_flux_files = {
    'U238': 'AntineutrinoSpectrum_238U.knt',
    'Th232': 'AntineutrinoSpectrum_232Th.knt',
}

geo_flux_normalizations = {
    'U238' : (2.7e3/5, 0.3),
    'Th232': (0.8e3/5, 0.3),
}

class Reactor(object):
    def __init__(self, name, location, fission_fractions,
                 power=None, power_rate=None):
        self.name = name
        self.location = location

        self.power = power

        self.power_rate = power_rate

        self.fission_fractions = {isoname: frac
                                  for isoname, frac in fission_fractions.iteritems()}

    def assign(self, ns):
        self.ns = ns("reactors")(self.name)
        self.ns.reqparameter("ThermalPower", central=self.power, sigma=0)

    def calcdistance(self, detector):
        return np.sqrt(np.sum((np.array(self.location) - np.array(detector.location))**2))

    def initdistance(self, ns, detector):
        dist = self.calcdistance(detector)
        pair_ns = detector.ns(self.name)
        pair_ns.reqparameter("L_{0}".format(self.name), central=dist, sigma=0)
        pair_ns.defparameter("L", target=pair_ns.ref("L_{0}".format(self.name)))

class ReactorGroup(object):
    def __init__(self, name, reactors):
        self.name = name
        self.reactors = reactors

        self.power = 1
        self.power_rate = reactors[0].power_rate

        self.fission_fractions = reactors[0].fission_fractions

    def assign(self, ns):
        self.ns = ns("reactorgroups")(self.name)

    def initdistance(self, ns, detector):
        pair_ns = detector.ns(self.name)
        bindings = {}
        for i, reactor in enumerate(self.reactors):
            bindings["L_{0}".format(i)] = detector.ns(reactor.name)["L"]
            bindings["P_{0}".format(i)] = reactor.ns["ThermalPower"]
        ROOT.ReactorGroup(len(self.reactors), ns=pair_ns, bindings=bindings)

        pair_ns.defparameter("L", target=pair_ns.ref("Lavg"))
        pair_ns.defparameter("ThermalPower", target=pair_ns.ref("Pavg"))

class Detector(object):
    def __init__(self, name, edges, location,
                 orders=None, protons=None, livetime=None, slicing=None, rtheta=None, npe=None):
        self.name = name
        self.edges = edges
        self.orders = orders

        self.location = location

        self.protons = protons

        self.livetime = livetime
        self.slicing = slicing
        self.rtheta = rtheta
        self.npe = npe

        self.unoscillated = None
        self.sumedspectra = None
        #self.eres_b = eres_b
        self.components = defaultdict(lambda: defaultdict(ROOT.Sum))
        self.backgrounds = defaultdict(ROOT.Sum)
        self.intermediates = {}
        self.intermediates_bkg = {}
        self.hists = defaultdict(dict)
        self.back_hists = defaultdict(dict)

    def load_correlation_matrix(self, weightns, weight_correlation):
        filename, histname = weight_correlation
        try:
            tfile = ROOT.TFile(filename, 'read')
            hist = tfile.Get(histname)
        except:
            tfile, hist = None, None

        if not hist:
            if tfile:
                tfile.ls()
            raise Exception('Unable to read histogram "%s" from file %s'%(histname, filename))

        from mpl_tools.root2numpy import get_buffer_hist2
        mat = get_buffer_hist2(hist)

        if mat.shape[0]!=mat.shape[1]:
            raise Exception('Non square matrix provided:', mat.shape[0], mat.shape[1])

        mmin, mmax = mat.min(), mat.max()
        if mmin<-1-1.e-12 or mmax>1.0+1.e-12:
            raise Exception('Invalid min/max correlation values:', mmin, mmax)

        pars = [par for (name,par) in weightns.walknames()]

        if len(pars)!=mat.shape[0]:
            raise Exception('Unable to set correlation to %i parameters with %ix%i matrix'%(len(pars, mat.shape[0], mat.shape[1])))

        from gna.parameters import covariance_helpers as ch
        ch.covariate_pars(pars, mat)

    def assign(self, ns, weight_correlation=None):
        self.ns = ns("detectors")(self.name)
        self.ns_weight = self.ns('weights')
        self.res_nses = []

        if self.protons is not None:
            self.ns.defparameter("TargetProtons", central=self.protons, sigma=0)

        self.ns.reqparameter("Eres_a", central=0.0, sigma=0)
        self.ns.reqparameter("Eres_c", central=0.0, sigma=0)

        if self.slicing:
            radius = self.slicing[-1]
            fiducial_volume = 4./3 * np.pi * radius**3

            def sphere_volume(r):
                return 4./3 * np.pi * r**3

            volumes = np.array([sphere_volume(x) for x in self.slicing])
            shell_volumes = volumes[1:] - volumes[:-1]
            shell_volumes = np.insert(shell_volumes,0,volumes[0])
            ratios = shell_volumes/fiducial_volume
            for idx, ratio in enumerate(ratios):
                self.ns_weight.defparameter("weight_{0}".format(str(idx)), central=ratio, sigma=0.)
                print('ratio {0} is {1}'.format(idx,ratio))
            if weight_correlation:
                self.load_correlation_matrix(self.ns_weight, weight_correlation)

            fin = TFile('input/npe-r-1mev.root')
            curve = fin.Get("curve")
            def cal_layer_res0(r):
                    return r*(curve.Eval(r))
            def cal_layer_res1(r):
                return r

            def cal_layer_res(mean_nn):
                return  1/np.sqrt(1/1.022*mean_nn)
                #return  1/np.sqrt(0.935828*mean_nn)
            #normalized center 1.42465
            #normalized medium 1.217484
            #normalized highest 1.168

            lista = np.insert(self.slicing,0,0)
            tmpp0 =  np.array([integrate.quad(cal_layer_res0,xx0,xx1) for xx0, xx1 in zip(lista[:-1], lista[1:])])
            tmpp1 =  np.array([integrate.quad(cal_layer_res1,xx0,xx1) for xx0, xx1 in zip(lista[:-1], lista[1:])])
            #centers = (lista[1:]+lista[:-1])*0.5
            tmpp =  tmpp0/tmpp1
            #tmpp2 = tmpp[:,0]
            #tmpp2 = [1189,1261,1295,1324,1349,1375,1400,1427,1458,1502]
            tmpp2 = [751,1154,1207,1229,1243,1255,1266,1275,1283,1291,1299,1306,1313,1320,1327,1333,1340,1346,1352,1358,1365,1371,1378,1384,1390,1397,1403,1410,1416,1423,1430,1438,1445,1453,1462,1472,1482,1494,1511,1537]
            eresbs = np.array([cal_layer_res(x) for x in tmpp2])
            for idx, eresb in enumerate(eresbs):
                ns_res = self.ns("res_{0}".format(idx))
                ns_res.defparameter("Eres_b", central=eresb, sigma=0.)
                self.res_nses.append(ns_res)
                print('layer {0} Eres_b is {1}'.format(idx,eresb))

        elif self.rtheta:
            lightyeild = self.rtheta[:]

            def cal_layer_res(mean_nn):
                return  1/np.sqrt(mean_nn)

            eresbs = np.array([cal_layer_res(x) for x in lightyeild])
            weight_rtheta = 1.0/len(lightyeild)
            weight_rtheta_error = 1.0/np.sqrt(131400/len(lightyeild))/200

            for idx, eresb in enumerate(eresbs):
                ns_res = self.ns("res_{0}".format(idx))
                ns_res.defparameter("Eres_b", central=eresb, sigma=0.)
                self.res_nses.append(ns_res)
                print('layer {0} Eres_b is {1}'.format(idx,eresb))
                self.ns_weight.defparameter("weight_{0}".format(str(idx)), central=weight_rtheta, sigma=weight_rtheta_error)
                print('ratio {0} is {1}'.format(idx,weight_rtheta))
            if weight_correlation:
                self.load_correlation_matrix(self.ns_weight, weight_correlation)
        else:
            self.ns.reqparameter("Eres_b", central=0.03, sigma=0)
        #  self.ns.reqparameter("rho_C14", central=1e-16, sigma=1e-16)

class GeoNeutrinoIsotope(object):
    def __init__(self, name):
        self.name = name

        try:
            Es_keV, self.ys = np.loadtxt(datapath(geo_flux_files[name]), unpack=True, skiprows=5)
        except OSError:
            raise Exception("Failed to load spectrum of {0} geo isotope from {1}".format(name, geo_flux_files[name]))
        self.Es = Es_keV*1e-3
        self.spectrum = ROOT.LinearInterpolator(len(self.Es), self.Es.copy(), self.ys.copy(), "use_zero")

class Isotope(object):
    def __init__(self, ns, name):
        self.ns = ns("isotopes")(name)
        self.name = name

        try:
            self.Es, self.ys = np.loadtxt(datapath(spectrumfiles[name]), unpack=True)
        except OSError:
            raise Exception("Failed to load spectrum of {0} reactor isotope from {1}".format(name, datapath(spectrumfiles[name])))

        self.spectrum = ROOT.LinearInterpolator(len(self.Es), self.Es.copy(), self.ys.copy(), "use_zero")

class ReactorExperimentModel(baseexp):
    oscprob_classes = {
            'standard': ROOT.OscProbPMNS,
            'decoh': ROOT.OscProbPMNSDecoh,
            }

    @classmethod
    def initparser(self, parser, env):
        """Initializes arguments parser with args common to all reactor experiments"""
        #  def OscProb(name):
            #  return {
                #  'standard': ROOT.OscProbPMNS,
                #  'decoh': ROOT.OscProbPMNSDecoh,
            #  }[name]
        parser.add_argument('--name', required=True)
        parser.add_argument('--ibd', choices=['zero', 'first'], default='zero')
        parser.add_argument('--backgrounds',choices=['geo'], action='append',
                            default=[], help='Choose backgrounds you want to add')
        parser.add_argument('--oscprob', choices=self.oscprob_classes.keys(),
                            default='standard')
        parser.add_argument('--binning', nargs=4, metavar=('DETECTOR', 'EMIN', 'EMAX', 'NBINS'),
                            action='append', default=[])
        parser.add_argument('--integration-order', type=int, default=4)
        parser.add_argument('--slicing', metavar='N', type=float, nargs='+',
                 help='radius of sliced detector layers')
        parser.add_argument('--rtheta', metavar='N', type=float, nargs='+',
                 help='mean Npe of each subdetector')
        parser.add_argument('--weight-correlation', nargs=2, metavar=("file.root", "name"), help='add correlation matrix for segment weights')
        parser.add_argument('--npe', action='store_true')
        #parser.add_argument('--eresb', metavar='N', type=float, nargs='+',
        #         help='eresb of sliced detector layers')
        parser.add_argument('--no-reactor-groups', action='store_true')
        parser.add_argument('--with-C14', action='store_true')

    def __init__(self, opts, ns=None, reactors=None, detectors=None):
        """Initialize a reactor experiment

        opts -- object with parsed common arguments returned by argparse
        ns -- namespace where experiment will create missing parameters, if is not provided ``opts.name`` will be used
        reactors -- iterable over Reactor objects, self.makereactors() will be called if None
        detectors -- iterable over Detector objects, self.makedetoctrs() will be called if None
        """
        super(ReactorExperimentModel, self).__init__(ns, opts)
        self._oscprobs = {}
        self._oscprobcls = self.oscprob_classes[self.opts.oscprob]
        self._isotopes = defaultdict(list)
        self._Enu_inputs = defaultdict(set)
        self.oscprobs_comps = defaultdict(dict)
        self._geo_isotopes = defaultdict(list)

        self.ns = ns or env.ns(opts.name)
        self.reqparameters(self.ns)

        self.detectors = list(detectors if detectors is not None else self.makedetectors())
        for binopt in self.opts.binning:
            for det in self.detectors:
                if det.name == binopt[0]:
                    break
            else:
                raise Exception("can't find detector {}".format(binopt[0]))
            det.edges = np.linspace(float(binopt[1]), float(binopt[2]), int(binopt[3]))
        for det in self.detectors:
            det.assign(self.ns, weight_correlation=self.opts.weight_correlation)
            if det.orders is None:
                det.orders = np.full(len(det.edges)-1, self.opts.integration_order, int)

        self.reactors = list(reactors if reactors is not None else self.makereactors())
        for reactor in self.reactors:
            reactor.assign(self.ns)

        self.linkpairs(self.reactors, self.detectors)

        for detector in self.detectors:
            self.make_backgrounds(detector, opts.backgrounds)
            self.setibd(detector, opts.ibd)
            self.setupobservations(detector)

    def make_backgrounds(self, detector, background_list):
        """Produce backgrounds and store them to detector components,
        currently only geoneutrino flux. Normalize it latter in makeibd"""

        #  pass
        for bkg in background_list:
            if bkg == 'geo':
                for iso_name in geo_flux_files.iterkeys():
                    geo_isotope = GeoNeutrinoIsotope(iso_name)
                    self._isotopes["geo_"+iso_name].append(geo_isotope)
                    self._Enu_inputs[detector].add(geo_isotope.spectrum.f.inputs.x)
                    detector.intermediates_bkg['geo_flux_{}'.format(iso_name)] = geo_isotope.spectrum



    def _getoscprobcls(self, reactor, detector):
        """Returns oscillation probability class for given (reactor, detector) pair"""
        if isinstance(reactor, ReactorGroup):
            return (ROOT.OscProbPMNSMult, ROOT.OscProbPMNS)
        keys = [frozenset([reactor, detector]), reactor, detector, None]
        for key in keys:
            try:
                return (self._oscprobs[key], self._oscprobs[key])
            except KeyError:
                pass
        return (self._oscprobcls, self._oscprobcls)

    def _getnormtype(self, reactor, detector):
        """Returns normalization type applicable for given reactor and detector

        if power, power rate, protons and livetime are provided, 'calc' is returned to calculate normalization
        otherwise 'manual' is returned meaning that normalization should be done by the caller with external Norm parameters
        """
        if all(x is not None for x in [reactor.power, reactor.power_rate, detector.protons, detector.livetime]):
            return 'calc'
        return 'manual'

    def groupreactors(self, reactors, detector, precision=1e-2):
        """Merges some Reactors of reactors to ReactorGroup if their distances to detector are similar enough

        Faster approximated formula is used. Only reactors with standard OscProbPMNS formula are considered for grouping.
        Approximation of the probability is of the order of order precision**3.
        """
        groups = [[reactor] for reactor in reactors]
        groupable = []
        if not self.opts.no_reactor_groups:
            for reactor in reactors:
                if self._getoscprobcls(reactor, detector)[0] is ROOT.OscProbPMNS:
                    if self._getnormtype(reactor, detector) == 'calc':
                        groupable.append(reactor)
                        groups.remove([reactor])
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
                rgroup = ReactorGroup("group{}".format(cnt), group)
                rgroup.assign(self.ns)
                rgroup.initdistance(self.ns, detector)
                grouped.append(rgroup)
                cnt += 1
            else:
                grouped.append(group[0])
        return grouped

    def normalization(self, reactor, detector, normtype):
        """Returns normalization object for given reactor/detector pair and norm type"""

        if normtype == 'calc':
            bindings = {}
            for isoname in reactor.fission_fractions:
                bindings["EnergyPerFission_{0}".format(isoname)] = self.ns("isotopes")(isoname)["EnergyPerFission"]
            norm = ROOT.ReactorNorm(vec(reactor.fission_fractions.keys()), bindings=bindings)
            norm.isotopes.livetime(C.Points(detector.livetime))
            norm.isotopes.power_rate(C.Points(reactor.power_rate))
            for isoname, frac in reactor.fission_fractions.iteritems():
                norm.isotopes['fission_fraction_{0}'.format(isoname)](C.Points(frac))
        elif normtype == 'manual':
            norm = ROOT.ReactorNormAbsolute(vec(reactor.fission_fractions.keys()))
            for isoname, frac in reactor.fission_fractions.iteritems():
                norm.isotopes['fission_fraction_{0}'.format(isoname)](frac)
        return norm

    def linkpairs(self, reactors, detectors):
        """Setup oscillations and normalizations of all reactors and detectors.

        Resulting components are stored in detector.components.
        """
        for reactor, detector in itertools.product(reactors, detectors):
            reactor.initdistance(self.ns, detector)

        for detector in detectors:
            detector.unoscillated = ROOT.Sum()
            grouped = self.groupreactors(reactors, detector)
            for rgroup in grouped:
                pair_ns = detector.ns(rgroup.name)
                normtype = self._getnormtype(rgroup, detector)
                with detector.ns, rgroup.ns, pair_ns:
                    norm = self.normalization(rgroup, detector, normtype)
                    oscprobcls, weightscls = self._getoscprobcls(rgroup, detector)
                    with self.ns("oscillation"):
                        oscprob = oscprobcls(ROOT.Neutrino.ae(), ROOT.Neutrino.ae())

                normedflux = ROOT.Sum()

                for isoname in rgroup.fission_fractions.keys():
                    isotope = Isotope(self.ns, isoname)
                    self._isotopes[(detector, rgroup)].append(isotope)
                    self._Enu_inputs[detector].add(isotope.spectrum.f.inputs.x)
                    subflux = ROOT.Product()
                    subflux.multiply(isotope.spectrum)
                    subflux.multiply(norm.isotopes['norm_{0}'.format(isoname)])
                    detector.intermediates['flux_{}'.format(isoname)] = subflux
                    normedflux.add(subflux)
                detector.intermediates['flux'] = normedflux


                compnames = set(oscprob.probsum.inputs)

                detector.unoscillated.add(normedflux)
                if 'comp0' in compnames:
                    detector.components['rate'][(weightscls, 'comp0')].add(normedflux)
                    ones = ROOT.FillLike(1.0)
                    ones.fill.inputs(normedflux)
                    detector.components['oscprob'][(weightscls, 'comp0')].add(ones)
                    self.oscprobs_comps[(detector, rgroup)][(weightscls, 'comp0')] = ones
                    compnames.remove('comp0')
                for osccomps in oscprob.transformations.itervalues():
                    for compname, osccomp in osccomps.outputs.iteritems():
                        if compname not in compnames:
                            continue
                        product = ROOT.Product()
                        product.multiply(normedflux)
                        product.multiply(osccomp)
                        detector.components['rate'][(weightscls, compname)].add(product)
                        detector.components['oscprob'][(weightscls, compname)].add(osccomp)
                        self.oscprobs_comps[(detector, rgroup)][(weightscls, compname)] = osccomp
                        if compname not in compnames:
                            raise Exception("overriden component {}".format(compname))
                        compnames.remove(compname)

                        if 'Enu' in osccomps.inputs:
                            self._Enu_inputs[detector].add(osccomps.inputs.Enu)
                if compnames:
                    raise Exception("components not found: {}".format(compnames))



    def setibd(self, detector, ibdtype):
        """Setup input energy values, integration and IBD calculation object for the given detector according to the ibdtype .
        ibdtype may be 'zero' or 'first' for the corresponding order.
        Resulting components are stored in detector.components.
        The backgrounds histos are also constructed here for now.
        """
        Evis_edges = detector.edges
        orders = detector.orders
        with self.ns("ibd"):
            econv = ROOT.EvisToEe()
        if ibdtype == 'zero':
            with self.ns("ibd"):
                ibd = ROOT.IbdZeroOrder()
            integrator = self.integrator = ROOT.GaussLegendre(Evis_edges, orders, len(orders))
            histcls = ROOT.GaussLegendreHist
            econv.Ee.Evis(integrator.points.x)
            ibd.xsec.Ee(econv.Ee.Ee)
            ibd.Enu.Ee(econv.Ee.Ee)
            eventsparts = [ibd.xsec]
        elif ibdtype == 'first':
            with self.ns("ibd"):
                ibd = ROOT.IbdFirstOrder()
            integrator = self.integrator = ROOT.GaussLegendre2d(Evis_edges, orders, len(orders), -1.0, 1.0, 5)
            histcls = ROOT.GaussLegendre2dHist
            econv.Ee.Evis(integrator.points.x)
            ibd.Enu.Ee(econv.Ee.Ee)
            ibd.Enu.ctheta(integrator.points.y)
            detector.intermediates['ctheta'] = integrator.points.y
            ibd.xsec.Enu(ibd.Enu)
            ibd.xsec.ctheta(integrator.points.y)
            ibd.jacobian.Enu(ibd.Enu)
            ibd.jacobian.Ee(integrator.points.x)
            ibd.jacobian.ctheta(integrator.points.y)
            eventsparts = [ibd.xsec, ibd.jacobian]
        else:
            raise Exception("unknown ibd type {0!r}".format(ibdtype))

        detector.intermediates['Enu'] = ibd.Enu
        detector.intermediates['xsec'] = ibd.xsec
        for inp in self._Enu_inputs.get(detector, []):
            inp.connect(ibd.Enu.Enu)


        for detector in self.detectors:
            for resname, comps in detector.components.iteritems():
                for compid, comp in comps.iteritems():
                    res = None
                    if resname == 'rate':
                        res = ROOT.Product()
                        res.multiply(comp)
                        for part in eventsparts:
                            res.multiply(part)
                    elif resname == 'oscprob':
                        res = comp
                    detector.hists[resname][compid] = histcls(integrator)
                    detector.hists[resname][compid].hist.inputs(res)

            # Below we construct backgrounds histos
            if self.opts.backgrounds:
                bkg_summary = ROOT.Sum()
                for bkg_name, bkg in detector.intermediates_bkg.iteritems():
                    prod = ROOT.Product()
                    prod.multiply(bkg)
                    unosc_bkg_name = 'unosc_' + bkg_name
                    for part in eventsparts:
                        prod.multiply(part)
                    detector.back_hists[unosc_bkg_name] = histcls(integrator)
                    detector.back_hists[unosc_bkg_name].hist.inputs(prod)

                    # normalize isotope flux now
                    iso_name = bkg_name.split('_')[-1]
                    with self.ns("geo_isotopes")(iso_name):
                        norm_geo = ROOT.GeoNeutrinoFluxNormed(detector.livetime[0]/year)
                    norm_geo.flux_norm.flux(detector.back_hists[unosc_bkg_name].hist)

                    with self.ns("oscillation"):
                        aver_oscprob = ROOT.OscProbAveraged(ROOT.Neutrino.ae(), ROOT.Neutrino.ae())
                    aver_oscprob.average_oscillations.inputs(norm_geo.flux_norm)
                    detector.back_hists['geo_'+iso_name] = norm_geo.flux_norm.normed_flux
                    bkg_summary.add(aver_oscprob.average_oscillations.flux_averaged_osc)




            detector.unoscillated_hist = histcls(integrator)
            res = ROOT.Product()
            res.multiply(detector.unoscillated)
            for part in eventsparts:
                res.multiply(part)
            detector.unoscillated_hist.hist.inputs(res)

            if self.opts.backgrounds:
                detector.unoscillated_with_bkg = ROOT.Sum()
                detector.unoscillated_with_bkg.add(bkg_summary)
                detector.unoscillated_with_bkg.add(detector.unoscillated_hist)
                detector.back_hists['sum_bkg'] = bkg_summary

    def _sumcomponents(self, components):
        oscprobs = {}
        for oscprobcls, compname in components.iterkeys():
            if oscprobcls in oscprobs:
                continue
            with self.ns("oscillation"):
                oscprob = oscprobcls(ROOT.Neutrino.ae(), ROOT.Neutrino.ae(),
                                     freevars=['L'])
            oscprobs[oscprobcls] = oscprob

        for compid, comp in components.iteritems():
            oscprobs[compid[0]].probsum[compid[1]](comp)

        probsums = [oscprob.probsum for oscprob in oscprobs.values()]
        if len(probsums) > 1:
            finalsum = ROOT.Sum()
            for probsum in probsums:
                finalsum.add(probsum)
            return finalsum
        else:
            return probsums[0]

    def setupobservations(self, detector):
        """Sum over the components and setup the observables to namespace"""

        self.ns.addobservable("{0}_unoscillated".format(detector.name), detector.unoscillated_hist.hist, export=False)
        if self.opts.backgrounds:
            self.ns.addobservable("{0}_unoscillated_with_bkg".format(detector.name),
                           detector.unoscillated_with_bkg, export=False)

        sums = {resname: self._sumcomponents(detector.hists[resname])
                for resname in detector.hists}

        if 'oscprob' in sums:
            detector.intermediates["oscprob"] = sums['oscprob']
        if 'rate' in sums:
            inter_sum = ROOT.Sum()
            sum_without_bkg = sums['rate']
            inter_sum.add(sum_without_bkg)
            if self.opts.backgrounds:
                inter_sum.add(detector.back_hists['sum_bkg'])

            self.ns.addobservable("{0}_noeffects".format(detector.name), inter_sum, export=False)

            if self.opts.with_C14:
                with self.ns('ibd'), detector.ns:
                    detector.c14 = ROOT.C14Spectrum(8,5)
                    detector.c14.smear.inputs(inter_sum)
                    finalsum = detector.c14.smear

                    self.ns.addobservable("{0}_c14".format(detector.name),
                                          finalsum, export=True)
            else:
                finalsum = inter_sum
                self.ns.addobservable("{0}_noc14".format(detector.name),finalsum, export=True)

            if detector.npe:
                with detector.ns:
                    npeq = ROOT.Npesmear()
                npeq.Npesmear.inputs(finalsum)
                self.ns.addobservable("weightedspectra", npeq.Npesmear)
                #with detector.ns:
                #    npeq = ROOT.EtoNpe()
                #try1points = C.Points(detector.edges)
                #npeq.etonpe.input(try1points)
                #try2points = npeq.etonpe
                #nlqua = ROOT.HistNonlinearity(True)
                #nlqua.set(self.integrator.points.xhist, try2points)
                #nlqua.add_input(finalsum)
                ##mat24 = nlqua.matrix.FakeMatrix.data()
                ##print(mat24)
                ##print(mat24.sum(axis=0))
                ##finalsum = nlqua.smear.Nrec
                #finalsumtmp = ROOT.Sum()
                #finalsumtmp.add(nlqua.smear.Nrec)
                #finalsum = finalsumtmp
                #self.ns.addobservable("{0}_qua".format(detector.name),
                #                     finalsum, export=True)
                #self.ns.addobservable("weightedspectra", npeq.Npesmear)

            elif detector.rtheta:
                with detector.ns_weight:
                    vnames=vec('weight_%i'%i for i in range(len(detector.rtheta)))
                    detector.sumedspectra = ROOT.WeightedSum(vnames, vnames)
                smeared_spectras = []
                for res_ns in detector.res_nses:
                    with res_ns, detector.ns:
                        eres = ROOT.EnergyResolution()
                        eres.matrix.Edges(self.integrator.points.xhist)
                        eres.smear.Ntrue(finalsum)
                        smeared_spectras.append(eres.smear.Nrec)

                for idx, eres in enumerate(smeared_spectras):
                    detector.sumedspectra.sum['weight_%i'%(idx)](eres)
                    self.ns.addobservable("subdetector{}".format(idx), eres)
                self.ns.addobservable("weightedspectra", detector.sumedspectra)

            elif detector.slicing:
                with detector.ns_weight:
                    vnames=vec('weight_%i'%i for i in range(len(detector.slicing)))
                    detector.sumedspectra = ROOT.WeightedSum(vnames, vnames)
                smeared_spectras = []
                for res_ns in detector.res_nses:
                    with res_ns, detector.ns:
                        eres = ROOT.EnergyResolution()
                        eres.matrix.Edges(self.integrator.points.xhist)
                        eres.smear.Ntrue(finalsum)
                        smeared_spectras.append(eres.smear.Nrec)

                for idx, eres in enumerate(smeared_spectras):
                    detector.sumedspectra.sum['weight_%i'%(idx)](eres)
                    self.ns.addobservable("layer{}".format(idx), eres)
                self.ns.addobservable("weightedspectra", detector.sumedspectra)

            else:
                with detector.ns:
                    orgeres = ROOT.EnergyResolution()
                orgeres.matrix.Edges(self.integrator.points.xhist)
                orgeres.smear.Ntrue(finalsum)
                self.ns.addobservable("weightedspectra", orgeres.smear.Nrec)

        det_ns = self.ns("detectors")(detector.name)

        for name in detector.intermediates:
            det_ns.addobservable(name, detector.intermediates[name], export=False)

        for name, bkg in detector.back_hists.iteritems():
            det_ns.addobservable("bkg_{}".format(name), bkg, export=False)

        for pair, comps in self.oscprobs_comps.iteritems():
            pair_ns = det_ns(pair[1].name)
            pair_ns.addobservable('oscprob', self._sumcomponents(comps), export=False)

    @classmethod
    def reqparameters(cls, ns):
        """Setup the parameters"""
        gna.parameters.ibd.reqparameters(ns("ibd"))
        gna.parameters.oscillation.reqparameters(ns("oscillation"))

        for isoname, (central, sigma) in eperfission.iteritems():
            isons = ns("isotopes")(isoname)
            isons.reqparameter("EnergyPerFission", central=central, sigma=sigma)
        for geo_isoname, (central, relsigma) in geo_flux_normalizations.iteritems():
            geo_isons = ns("geo_isotopes")(geo_isoname)
            geo_isons.reqparameter("FluxNorm", central=central, relsigma=sigma)
