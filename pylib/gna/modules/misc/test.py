from gna.ui import basecmd
import ROOT
import numpy as np
import physlib

class test(basecmd):
    def run(self):
        env = self.env.default
        pars = env.pars
        pars.define('SinSq12', central=0.307, sigma=0.017, limits=(0, 1))
        pars.define('DeltaMSq12', central=7.54e-5, sigma=0.24e-5,
                    limits=(0, 0.1))
        pars.define('L', central=52, sigma=0)

        pdg = physlib.pdg[2012]
        pars.define("NeutronLifeTime", central=pdg['neutron_lifetime'],
                    sigma=0)
        pars.define("ProtonMass", central=pdg['ProtonMass'], sigma=0)
        pars.define("NeutronMass", central=pdg['NeutronMass'], sigma=0)
        pars.define("ElectronMass", central=pdg['ElectronMass'], sigma=0)

        prediction = ROOT.PredictionSet()
        edges = np.array([1, 2, 3, 4, 5], dtype=float)
        orders = np.array([5]*(len(edges)-1), dtype=int)
        integrator = ROOT.GaussLegendre(edges, orders, len(orders))
        events = ROOT.Product()
        ibd = ROOT.IbdZeroOrder()
        ibd["Enu"].inputs[0].connect(integrator["points"].outputs[0])
        ibd["xsec"].inputs[0].connect(ibd["Enu"].outputs[0])
        events.add(ibd["xsec"].outputs[0])
        oscprob = ROOT.OscProb2nu()
        oscprob["prob"].inputs[0].connect(ibd["Enu"].outputs[0])
        events.add(oscprob["prob"].outputs[0])

        # integrator = ROOT.GaussLegendre2d(edges, orders, len(orders), -1.0, 1.0, 5)
        # events = ROOT.Product()
        # ibd = ROOT.IbdFirstOrder()
        # ibd["Enu"].inputs[0].connect(integrator["points"].outputs[0])
        # ibd["Enu"].inputs[1].connect(integrator["points"].outputs[1])
        # ibd["xsec"].inputs[0].connect(ibd["Enu"].outputs[0])
        # ibd["xsec"].inputs[1].connect(integrator["points"].outputs[1]);
        # events.add(ibd["xsec"].outputs[0])
        # ibd["jacobian"].inputs[0].connect(ibd["Enu"].outputs[0])
        # ibd["jacobian"].inputs[1].connect(integrator["points"].outputs[0])
        # ibd["jacobian"].inputs[2].connect(integrator["points"].outputs[1])
        # events.add(ibd["jacobian"].outputs[0])
        # oscprob = ROOT.OscProb2nu()
        # oscprob["prob"].inputs[0].connect(ibd["Enu"].outputs[0])
        # events.add(oscprob["prob"].outputs[0])

        integrator["hist"].inputs[0].connect(events["product"].outputs[0])
        prediction.add(integrator["hist"].outputs[0])
        # prediction.add(ibd["xsec"].outputs[0])
        print np.frombuffer(prediction.data(), dtype=float, count=4)
