# -*- coding: utf-8 -*-
from __future__ import absolute_import
import physlib

def reqparameters(ns, pdg_year=None):
    if pdg_year is not None:
        pdg = physlib.pdg[pdg_year]
    else:
        pdg = physlib.pdg[2012]

    ns.reqparameter("NeutronLifeTime", central=pdg['neutron_lifetime'], sigma=0, label='Neutron life time, s')
    ns.reqparameter("ProtonMass", central=pdg['ProtonMass'], sigma=0, label='Proton mass, MeV')
    ns.reqparameter("NeutronMass", central=pdg['NeutronMass'], sigma=0, label='Neutron mass, MeV')
    ns.reqparameter("ElectronMass", central=pdg['ElectronMass'], sigma=0, label='Electron mass, MeV')
