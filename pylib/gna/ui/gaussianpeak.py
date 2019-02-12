from gna.ui import basecmd
from gna.env import env
import ROOT
import numpy as np

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--name',   required=True, halp='observation name')
        parser.add_argument('--npeaks', default=1,   type=int, help='number of peaks')
        parser.add_argument('--Emin',   default=0,   type=float, help='Minimal energy')
        parser.add_argument('--Emax',   default=5,   type=float, help='Maximal Energy')
        parser.add_argument('--nbins',  default=100, type=int, help='Number of bins')
        parser.add_argument('--order',  default=8, help='Order of integrator for each bin (Gauss-Legendre)')
        parser.add_argument('--no-eres', action='store_false', dest='with_eres', help='Disable energy resoulution')

    def init(self):
        if self.opts.npeaks == 1:
            names = [self.opts.name]
        else:
            names = [self.opts.name + str(i) for i in range(self.opts.npeaks)]
        ns = env.ns(self.opts.name)
        ns.reqparameter("Eres_a", central=0.0, sigma=0)
        ns.reqparameter("Eres_b", central=0.03, sigma=0)
        ns.reqparameter("Eres_c", central=0.0, sigma=0)

        peak_sum = ROOT.Sum(labels='Sum of\nsignals')
        common_ns = env.ns(self.opts.name)
        edges = np.linspace(self.opts.Emin, self.opts.Emax, self.opts.nbins+1)
        orders = np.array([self.opts.order]*(len(edges)-1), dtype=int)

        integrator = ROOT.GaussLegendre(edges, orders, len(orders), labels='GL sampler')
        for i, name in enumerate(names):
            locns = env.ns(name)
            locns.reqparameter('BackgroundRate', central=50, relsigma=0.1)
            locns.reqparameter('Mu', central=100, relsigma=0.1)
            locns.reqparameter('E0', central=2, sigma=0.05)
            locns.reqparameter('Width', central=0.2, sigma=0.005)
            with locns:
                model = ROOT.GaussianPeakWithBackground(labels='Peak %i'%i)

            model.rate.E(integrator.points.x)
            hist = ROOT.GaussLegendreHist(integrator, labels='Integrator')
            hist.hist.f(model.rate.rate)
            peak_sum.add(hist.hist)
            locns.addobservable('spectrum', hist.hist)

        common_ns.addobservable('spectrum', peak_sum)
        with ns:
             eres = ROOT.EnergyResolutionC(labels='Energy\nresolution')
        eres.smear.inputs(peak_sum)
        ns.addobservable("spectrum_with_eres", eres.smear)

        ns.printparameters(labels='True')
