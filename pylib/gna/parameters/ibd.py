import physlib

def defparameters(ns):
    pdg = physlib.pdg[2012]
    ns.defparameter("NeutronLifeTime", central=pdg['neutron_lifetime'], sigma=0)
    ns.defparameter("ProtonMass", central=pdg['ProtonMass'], sigma=0)
    ns.defparameter("NeutronMass", central=pdg['NeutronMass'], sigma=0)
    ns.defparameter("ElectronMass", central=pdg['ElectronMass'], sigma=0)
