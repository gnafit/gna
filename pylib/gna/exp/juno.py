from gna.exp import baseexp
import gna.exp.reactor
import numpy as np
import ROOT

class exp(baseexp):
    @classmethod
    def initparser(self, parser):
        parser.add_argument('--ns', default='juno')
        parser.add_argument('--ibd', choices=['zero', 'first'], default='zero')
        parser.add_argument('--erange', type=float, nargs=2,
                            default=[1.0, 8.0],
                            metavar=('E_MIN', 'E_MAX'),
                            help='energy range')
        parser.add_argument('--nbins', type=int,
                            default=200,
                            help='number of bins')

    def makereactors(self):
        common = {
            'power_rate': [1.0],
            'isotopes': self.isotopes,
            'fission_fractions': {
                'Pu239': [0.60],
                'Pu241': [0.07],
                'U235': [0.27],
                'U238': [0.06],
            }
        }
        data = [
            {'name': 'YJ1', 'location': 52.75, 'power': 2.9},
            {'name': 'YJ2', 'location': 52.84, 'power': 2.9},
            {'name': 'YJ3', 'location': 52.42, 'power': 2.9},
            {'name': 'YJ4', 'location': 52.51, 'power': 2.9},
            {'name': 'YJ5', 'location': 52.12, 'power': 2.9},
            {'name': 'YJ6', 'location': 52.21, 'power': 2.9},

            {'name': 'TS1', 'location': 52.76, 'power': 4.6},
            {'name': 'TS2', 'location': 52.63, 'power': 4.6},
            {'name': 'TS3', 'location': 52.32, 'power': 4.6},
            {'name': 'TS4', 'location': 52.20, 'power': 4.6},

            {'name': 'DYB', 'location': 215.0, 'power': 17.4},
            {'name': 'HZ', 'location': 265.0, 'power': 17.4},
        ]
        return gna.exp.reactor.makereactors(self.ns, common, data)

    def makedetectors(self):
        common = {
            'livetime': [6*365*24*60*60.0],
        }
        data = [
            {'name': 'AD1', 'location': .0, 'protons': 1.42e33},
        ]
        return gna.exp.reactor.makedetectors(self.ns, common, data)

    def __init__(self, env, opts):
        self.opts = opts
        self.ns = env.ns(opts.ns)
        gna.exp.reactor.defparameters(self.ns)

        edges = np.linspace(opts.erange[0], opts.erange[1], opts.nbins+1)
        orders = np.array([10]*(len(edges)-1), dtype=int)

        Enu, compfactory = gna.exp.reactor.init(self.ns, edges, orders, opts.ibd)

        self.isotopes = gna.exp.reactor.makeisotopes(self.ns)
        self.reactors = self.makereactors()
        self.detectors = self.makedetectors()

        for isotope in self.isotopes:
            isotope.spectrum.f.inputs(Enu)

        gna.exp.reactor.setupcomponents(self.ns, self.reactors, self.detectors, Enu)
        for detector in self.detectors:
            gna.exp.reactor.setupobservations(self.ns, detector, compfactory)
