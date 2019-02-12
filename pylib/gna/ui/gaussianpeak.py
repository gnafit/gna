from __future__ import print_function
from gna.ui import basecmd
from gna.env import env
import ROOT
import numpy as np
from gna import constructors as C

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--name',   required=True, help='observation name')
        parser.add_argument('--npeaks', default=1,   type=int, help='number of peaks')
        parser.add_argument('--Emin',   default=0,   type=float, help='Minimal energy')
        parser.add_argument('--Emax',   default=5,   type=float, help='Maximal Energy')
        parser.add_argument('--nbins',  default=100, type=int, help='Number of bins')
        parser.add_argument('--order',  default=8, type=int, help='Order of integrator for each bin (Gauss-Legendre)')
        parser.add_argument('--with-eres', '--eres', action='store_true', help='Enable energy resoulution')

    def init(self):
        if self.opts.npeaks == 1:
            names = [self.opts.name]
        else:
            names = [self.opts.name + str(i) for i in range(self.opts.npeaks)]
        common_ns = env.ns(self.opts.name)

        if self.opts.with_eres:
            common_ns.reqparameter("Eres_a", central=0.0, sigma=0)
            common_ns.reqparameter("Eres_b", central=0.03, sigma=0)
            common_ns.reqparameter("Eres_c", central=0.0, sigma=0)

        peak_sum = ROOT.Sum(labels='Sum of\nsignals')
        edges = np.linspace(self.opts.Emin, self.opts.Emax, self.opts.nbins+1, dtype='d')

        integrator = C.IntegratorGL(edges, self.opts.order, labels=('GL sampler', 'GL integrator'))
        for i, name in enumerate(names):
            locns = env.ns(name)
            locns.reqparameter('BackgroundRate', central=50, relsigma=0.1)
            locns.reqparameter('Mu', central=100, relsigma=0.1)
            locns.reqparameter('E0', central=2, sigma=0.05)
            locns.reqparameter('Width', central=0.2, sigma=0.005)
            with locns:
                model = ROOT.GaussianPeakWithBackground(labels='Peak %i'%i)

            model.rate.E(integrator.points.x)
            if i:
                integrator.add_transformation()
            print(i)
            integrator.print()
            out = integrator.add_input(model.rate.rate)
            peak_sum.add(out)
            locns.addobservable('spectrum', out)

        common_ns.addobservable('spectrum', peak_sum)

        if self.opts.with_eres:
            with common_ns:
                 eres = ROOT.EnergyResolution(True, labels='Energy\nresolution')
            peak_sum.sum >> eres.matrix.Edges
            peak_sum.sum >> eres.smear.Ntrue
            common_ns.addobservable("spectrum_with_eres", eres.smear.Nrec)

        env.globalns.printparameters(labels='True')
