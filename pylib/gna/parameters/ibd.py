import physlib

def reqparameters(ns):
    pdg = physlib.pdg[2012]
    ns.reqparameter("NeutronLifeTime", central=pdg['neutron_lifetime'], sigma=0)
    ns.reqparameter("ProtonMass", central=pdg['ProtonMass'], sigma=0)
    ns.reqparameter("NeutronMass", central=pdg['NeutronMass'], sigma=0)
    ns.reqparameter("ElectronMass", central=pdg['ElectronMass'], sigma=0)
