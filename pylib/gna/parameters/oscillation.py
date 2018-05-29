#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ROOT
from gna.env import env
from physlib import pdg
curpdg = pdg[2016]


# TODO: Add the way to automaticaly detect the current mass ordering and
# switch the values of mixings to the current known global best fit for a
# given ordering
# For a given moment all parameters are just set for normal ordering

# DeltaMSqIJ = m^2_j - m^2_i

def reqparameters(ns):
    ns.reqparameter('SinSq12', central=curpdg['sinSqtheta12'],
                      sigma=curpdg['sinSqtheta12_e'], limits=(0,1), label='Solar mixing angle amplitude sin²θ₁₂')

    ns.reqparameter('DeltaMSq12',central=curpdg['dmSq21'],
                      sigma=curpdg['dmSq21_e'], limits=(0, 0.1), label='Solar mass splitting Δm²₂₁')

    ns.reqparameter('SinSq13', central=curpdg['sinSqtheta13'],
                      sigma=curpdg['sinSqtheta13_e'], limits=(0,1), label='Reactor mixing angle amplitude sin²θ₁₃ ')

    ns.reqparameter('Alpha', type='discrete', default='normal',
            variants={'normal': 1.0, 'inverted': -1.0}) #, label='Neutrino mass hierarchy: 1 (NH) or -1 (IH)'

    ns.reqparameter('SinSq23', central=curpdg['sinSqtheta23_normal'],
                      sigma=curpdg['sinSqtheta23_normal_e'], limits=(0,1), label='Atmospheric mixing angle amplitude sin²θ₂₃')

    #  ns.reqparameter('DeltaMSq23', central=curpdg['dmSq32_normal'],
                     #  sigma=curpdg['dmSq32_normal_e'], limits=(0, 0.1))

    ns.reqparameter('DeltaMSqEE', central=curpdg['dmSqEE'],
                      sigma=curpdg['dmSqEE_e'], limits=(0, 0.1), label='Reactor average mass splitting Δm²(ee)')
    ns.reqparameter('Delta', type='uniformangle', central=0.0, label='CP violation phase δ(CP)')
    ns.reqparameter("SigmaDecohRel", central=1.e-5, sigma=1e-5, label='Relative momentum spread (decoherence)')

    with ns:
        ROOT.OscillationExpressions(ns=ns)
        ROOT.PMNSExpressions(ns=ns)

#  def reqparameters(ns):
    #  ns.reqparameter('SinSq12', central=0.307, sigma=0.017, limits=(0, 1))
    #  ns.reqparameter('SinSq13', central=0.0214, sigma=0.002, limits=(0, 1))
    #  ns.reqparameter('SinSq23', central=0.42, sigma=0.08, limits=(0, 1))

    #  ns.reqparameter('DeltaMSq12', central=7.54e-5, sigma=0.24e-5,
                    #  limits=(0, 0.1))
    #  ns.reqparameter('DeltaMSqEE', central=2.34e-3, sigma=0.10e-3,
                    #  limits=(0, 0.1))
    #  ns.reqparameter('Alpha', type='discrete', default='normal',
                    #  variants={'normal': 1.0, 'inverted': -1.0})
    #  ns.reqparameter('Delta', type='uniformangle', central=0.0)
    #  ns.reqparameter("SigmaDecohRel", central=1.e-5, sigma=0)

    #  with ns:
        #  ROOT.OscillationExpressions(ns=ns)
        #  ROOT.PMNSExpressions(ns=ns)

