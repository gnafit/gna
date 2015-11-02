from gna.ui import basecmd
import physlib
import ROOT

class ibd(basecmd):
    def init(self):
        env = self.env

        pdg = physlib.pdg[2012]
        env.defparameter("NeutronLifeTime", central=pdg['neutron_lifetime'],
                         sigma=0)
        env.defparameter("ProtonMass", central=pdg['ProtonMass'], sigma=0)
        env.defparameter("NeutronMass", central=pdg['NeutronMass'], sigma=0)
        env.defparameter("ElectronMass", central=pdg['ElectronMass'], sigma=0)
