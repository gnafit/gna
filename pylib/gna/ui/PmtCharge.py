# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from gna.ui import basecmd
from gna.env import env
import ROOT
import numpy as np
import scipy.misc
from scipy.stats import poisson
import gna.constructors as C

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--name', required=True)
        parser.add_argument('--Emin', default=0, type=float)
        parser.add_argument('--Emax', default=5, type=float)
        parser.add_argument('--nbins', default=70, type=int)
        parser.add_argument('--order', default=8)
        parser.add_argument('--PoissonMean', default=0.2, type=float)
        parser.add_argument('--PoissonOrder', default=4, type=int)

    def init(self):
        ns = env.ns(self.opts.name)
        ns.reqparameter('BackgroundRate', central=0, sigma=0.1)
        ns.reqparameter('Mu', central=1, sigma=1)
        ns.reqparameter('E0', central=2, sigma=0.05)
        ns.reqparameter('Width', central=0.2, sigma=0.005)

        edges = np.linspace(self.opts.Emin, self.opts.Emax, self.opts.nbins+1)
        orders = np.array([self.opts.order]*(len(edges)-1), dtype=int)
        integrator = ROOT.GaussLegendre(edges, orders, len(orders))
        hist = ROOT.GaussLegendreHist(integrator)
        signal = ROOT.Sum()

        n = self.opts.PoissonOrder
        model = {}
        #ff = np.arange(1,n+1)
        #ff = 1/scipy.misc.factorial(ff)*np.exp(-self.opts.PoissionMean)
        #ff_points = C.Points(ff)
        #print(ff, ff_points)
        with ns:
            for i in range(1, n+1):
                print(i, n)
                model[i] =  ROOT.GaussianPeakWithBackground(i)
                model[i].rate.E(integrator.points.x)
         #       print(model[i].rate,model[i].rate.rate,ff_points[i])
                prod = ROOT.Product()
                prod.multiply(model[i].rate.rate)
                poisson_factor = poisson.pmf(i, self.opts.PoissonMean)
                poisson_factor_prod = C.Points([poisson_factor])
                print(type(model[i].rate), poisson_factor, poisson_factor_prod)
                prod.multiply(poisson_factor_prod)
                signal.add(prod)
        hist.hist.f(signal)
        ns.addobservable('spectrum', hist.hist)
