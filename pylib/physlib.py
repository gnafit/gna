#!/usr/bin/env python

# Here we might collect some useful common things
from math import  pow, sqrt

#  TODO: how to correctly include hierarchy dependent \Delta m^2_{32} update?
#  Before the MINOS 2011 was used
use_pdg_version = 2016

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

pdg[2016]  = dict( pdg[2013]
    , NeutronMass = 939.565413
    , ProtonMass  = 938.272081
    , ElectronMass = 0.510998946
    , neutron_lifetime = 880.2
    , neutron_lifetimee = 1.0
    , dmSq21 = 7.53e-5 #pdglive, 2013, Gando (KamLAND + solar + SBL + accelerator)
    , dmSq21_e = 0.18e-5
    , sinSqtheta12 = 0.304
    , sinSqtheta12_e = 0.014 # the error in pdg is asymmetric, took the upper bound
    , sinSqtheta13 = 2.19e-2
    , sinSqtheta13_e = 0.12e-2
    , dmSq32_normal = 2.44e-3 # pdglive, 2016, their own fit
    , dmSq32_normal_e = 0.06e-3
    , dmSq32_inverted = 2.51e-3
    , dmSq32_inverted_e = 0.06e-3
    , sinSqtheta23_normal = 0.50 # pdglive, 2016, their own fit
    , sinSqtheta23_normal_e = 0.05
    , sinSqtheta23_inverted = 0.51
    , sinSqtheta23_inverted_e = 0.05
    , dmSqEE = 2.5e-3  # Daya Bay long oscillation paper
    , dmSqEE_e = 0.8485e-3
)

pdg[2018]  = dict( pdg[2016]
    , dmSq21 = 7.53e-5 #pdglive, 2013, Gando (KamLAND + solar + SBL + accelerator)
    , dmSq21_e = 0.18e-5
    , sinSqtheta12 = 0.307
    , sinSqtheta12_e = 0.013
    , sinSqtheta13 = 2.12e-2
    , sinSqtheta13_e = 0.08e-2
    , dmSq32_normal = 2.51e-3
    , dmSq32_normal_e = 0.05e-3
    , dmSq32_inverted = 2.56e-3
    , dmSq32_inverted_e = 0.04e-3
    , dmSqEE = 0.0024924088
)

pdg['dyboscar']  = dict( pdg[2013]
    , NeutronMass = 939.565379
    , ProtonMass  = 938.272046
    , ElectronMass = 0.510998928
    , neutron_lifetime = 880.3
    , neutron_lifetimee = 1.1
    , dmSq21 = 7.53e-5
    , dmSq21_e = 0.18e-5
    #
    , dmSq32 = 2.45e-3
    , dmSq32_e = 0.02e-3
    #
    , sinSq2theta12 = 0.846
    , sinSq2theta12_e = 0.021
    , sinSqtheta12 = 0.304
    , sinSqtheta12_e = 0.013
    #
    , sinSq2theta13 = 0.084
    , sinSq2theta13_e = 0.002
    , sinSqtheta13 = (1 - sqrt(1-0.084))/2
    , sinSqtheta13_e = 2e-3

    , sinSqtheta23_normal = 0.514 # pdglive, 2016, their own fit
    , sinSqtheta23_normal_e = 0.055
    , sinSqtheta23_inverted = 0.51
    , sinSqtheta23_inverted_e = 0.055
)

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

def addDoubleTheta(year, varname, verbose=False):
    d = pdg[year]
    assert varname.startswith('sinSqtheta')

    errname = varname+'_e'

    central = d[varname]
    sigma = d[errname]

    left, right = central-sigma, central+sigma

    newname = varname.replace('sinSq', 'sinSq2')
    newerrname = newname+'_e'

    def cnv(v):
        return 4.0*v*(1.0-v)

    newcentral = cnv(central)
    newleft = cnv(left)
    newright = cnv(right)

    newerr = (newright-newleft)*0.5
    asymm  = (newright+newleft)*0.5 - newcentral

    d[newname] = newcentral
    d[newerrname]  = newerr

    if verbose:
        print(u'[{year}] Add evaluated {name}={central}±{err}. Asymmetry: {asymm}'.format(year=year, name=newname, central=newcentral, err=newerr, asymm=asymm))

addDoubleTheta(2016, 'sinSqtheta13')
addDoubleTheta(2016, 'sinSqtheta12')
addDoubleTheta(2018, 'sinSqtheta13')
addDoubleTheta(2018, 'sinSqtheta12')

pc = PhysicsConstants()
percent = 0.01
