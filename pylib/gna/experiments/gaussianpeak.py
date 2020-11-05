"""Example model: gaussian peak with flat background"""

from gna.exp import baseexp
from gna.env import env
import ROOT
import numpy as np
from gna import constructors as C

class exp(baseexp):
    """Example model: gaussian peak with flat background"""

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--npeaks', default=1,   type=int, help='number of peaks')
        parser.add_argument('--Emin',   default=0,   type=float, help='Minimal energy')
        parser.add_argument('--Emax',   default=5,   type=float, help='Maximal Energy')
        parser.add_argument('--nbins',  default=200, type=int, help='Number of bins')
        parser.add_argument('--order',  default=8, type=int, help='Order of integrator for each bin (Gauss-Legendre)')
        parser.add_argument('--with-eres', '--eres', action='store_true', help='Enable energy resoulution')
        parser.add_argument('--print', action='store_true', help='Print parameters')

    def init(self):
        if self.opts.npeaks == 1:
            names = ['peak']
            peak_sum = None
        else:
            names = ['peak_%i'%(i) for i in range(self.opts.npeaks)]
            peak_sum = ROOT.Sum(labels='Sum of\nsignals')

        if self.opts.with_eres:
            self.namespace.reqparameter("Eres_a", central=0.0, sigma=0)
            self.namespace.reqparameter("Eres_b", central=0.03, sigma=0)
            self.namespace.reqparameter("Eres_c", central=0.0, sigma=0)

        edges = np.linspace(self.opts.Emin, self.opts.Emax, self.opts.nbins+1, dtype='d')

        integrator = C.IntegratorGL(edges, self.opts.order, labels=('GL sampler', 'GL integrator'))
        for i, name in enumerate(names):
            locns = env.ns(name)
            locns.reqparameter('BackgroundRate', central=50, relsigma=0.1, label='Flat background rate %i'%i)
            locns.reqparameter('Mu', central=100, relsigma=0.1, label='Peak %i amplitude'%i)
            locns.reqparameter('E0', central=2, sigma=0.05, label='Peak %i position'%i)
            locns.reqparameter('Width', central=0.2, sigma=0.005, label='Peak %i width'%i)
            with locns:
                model = ROOT.GaussianPeakWithBackground(labels='Peak %i'%i)

            model.rate.E(integrator.points.x)
            if i:
                integrator.add_transformation()
            out = integrator.add_input(model.rate.rate)
            if peak_sum:
                peak_sum.add(out)
                locns.addobservable('spectrum', out)
            else:
                peak_sum=out

        self.namespace.addobservable('spectrum', peak_sum)

        if self.opts.with_eres:
            with self.namespace:
                 eres = ROOT.EnergyResolution(True, labels='Energy\nresolution')
            peak_sum.sum >> eres.matrix.Edges
            peak_sum.sum >> eres.smear.Ntrue
            self.namespace.addobservable("spectrum_with_eres", eres.smear.Nrec)

        if self.opts.print:
            self.namespace.printparameters(labels='True')
