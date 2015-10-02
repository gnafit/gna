#!/usr/bin/env python
# encoding: utf-8

# Here we might collect some useful common things
from math import  pow

use_pdg_version = 2012

pdg = {}
pdg[2010] = dict(
      sinSq2theta12 = 0.861
    , sinSq2theta12e = [ 0.022, 0.026 ] # lower, upper
    #
    , dmSq21 = 7.59e-5
    , dmSq21e = 0.21e-5
    #
    , dmSq32 = 2.43e-3
    , dmSq32e = 0.13e-3
    #
    , neutron_lifetime = 885.7
    , neutron_lifetimee = 0.8
    #
    , ProtonMass  = 938.272013
    , NeutronMass = 939.565346
    , ElectronMass = 0.51099891
)
pdg[2011] = dict( pdg[2010] # inherit PDG2010
    , sinSq2theta12 = 0.861
    , sinSq2theta12e = [ 0.022, 0.026 ] # lower, upper
    #
    , dmSq21 = 7.59e-5
    , dmSq21e = 0.21e-5
    #
    , dmSq32 = 2.43e-3
    , dmSq32e = 0.13e-3
    #
    , neutron_lifetime = 881.5
    , neutron_lifetimee = 1.5
)
pdg[2012] = dict( pdg[2011] # inherit pdg[2011]
    , sinSq2theta12 = 0.857
    , sinSq2theta12e = 0.024
    #
    , dmSq21 = 7.50e-5 # pdglive, 2011, gando (kamland+solar)
    , dmSq21e = 0.20e-5
    #
    , dmSq32 = 2.32e-3 # pdglive, 2011, ADAMSON (minos)
    , dmSq32e = [ 0.08e-3, 0.12e-3 ] #lower upper
    #
    , neutron_lifetime = 880.1
    , neutron_lifetimee = 1.1
    #
    , ProtonMass  = 938.272046
    , NeutronMass = 939.565379
    , ElectronMass = 0.510998928
)
pdg[2013]   = dict( pdg[2012]
    , neutron_lifetime = 880.1
    , neutron_lifetimee = 0.9
)
pdg['live'] = dict( pdg[2013] )

class PhysicsConstants:
    def __init__(self, pdgver = use_pdg_version):
        self.__dict__.update( pdg[pdgver] )

        self.DeltaNP     = self.NeutronMass - self.ProtonMass
        self.NucleonMass = (self.NeutronMass + self.ProtonMass)/2

        self.ElectronMass2 = self.ElectronMass**2
        self.NeutronMass2  = self.NeutronMass**2
        self.ProtonMass2   = self.ProtonMass**2
    ##end def function__init__
##end class PhysicsConstants

pc = PhysicsConstants()



