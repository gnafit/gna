from gna.ui import basecmd
import ROOT
import numpy as np
import physlib

class test(basecmd):
    def run(self):
        env = self.env.default
        pars = env.pars
        pars.define('SinSq12', central=0.307, sigma=0.017, limits=(0, 1))
        pars.define('SinSq13', central=0.0214, sigma=0.002, limits=(0, 1))
        pars.define('SinSq23', central=0.42, sigma=0.08, limits=(0, 1))

        pars.define('DeltaMSq12', central=7.54e-5, sigma=0.24e-5,
                    limits=(0, 0.1))
        pars.define('DeltaMSqEE', central=2.44e-3, sigma=0.10e-3,
                    limits=(0, 0.1))
        pars.define('Alpha', type='discrete', default='normal',
                    variants={'normal': 1.0, 'inverted': -1.0})

        pars.define('Delta', type='uniformangle', central=0.0)

        pars.define('L', central=2, sigma=0)

        pdg = physlib.pdg[2012]
        pars.define("NeutronLifeTime", central=pdg['neutron_lifetime'],
                    sigma=0)
        pars.define("ProtonMass", central=pdg['ProtonMass'], sigma=0)
        pars.define("NeutronMass", central=pdg['NeutronMass'], sigma=0)
        pars.define("ElectronMass", central=pdg['ElectronMass'], sigma=0)

        expressions = []
        expressions.append(ROOT.OscillationExpressions())
        expressions.append(ROOT.PMNSExpressions())

        edges = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        orders = np.array([5]*(len(edges)-1), dtype=int)

        prediction = ROOT.PredictionSet()
        integrator = ROOT.GaussLegendre(edges, orders, len(orders))
        events = ROOT.Product()
        ibd = ROOT.IbdZeroOrder()
        ibd.Enu.inputs(integrator.points)
        ibd.xsec.inputs(integrator.points)
        events.multiply(ibd.xsec)
        oscprob = ROOT.OscProb2nu()
        oscprob.prob.inputs(ibd.Enu)
        events.multiply(oscprob.prob)
        integrator.hist.inputs(events.product)
        prediction.append(integrator.hist)
        ibd0 = np.frombuffer(prediction.data(), dtype=float, count=prediction.size()).copy()
        print ibd0

        prediction = ROOT.PredictionSet()
        integrator = ROOT.GaussLegendre2d(edges, orders, len(orders), -1.0, 1.0, 5)
        events = ROOT.Product()
        ibd = ROOT.IbdFirstOrder()
        ibd.Enu.inputs(integrator.points)
        ibd.xsec.Enu(ibd.Enu)
        ibd.xsec.ctheta(integrator.points.y)
        events.multiply(ibd.xsec)
        ibd.jacobian.Enu(ibd.Enu)
        ibd.jacobian.Ee(integrator.points.x)
        ibd.jacobian.ctheta(integrator.points.y)
        events.multiply(ibd.jacobian)
        oscprob = ROOT.OscProb2nu()
        oscprob.prob.inputs(ibd.Enu)
        events.multiply(oscprob.prob)
        integrator.hist.inputs(events.product)
        prediction.append(integrator.hist)
        ibd1 = np.frombuffer(prediction.data(), dtype=float, count=prediction.size()).copy()
        print ibd1

        print ibd1/ibd0
