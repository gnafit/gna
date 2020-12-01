#!/usr/bin/env python

import ROOT
from gna.env import env
from physlib import pdg
from gna import constructors as C


# TODO: Add the way to automaticaly detect the current mass ordering and
# switch the values of mixings to the current known global best fit for a
# given ordering
# For a given moment all parameters are just set for normal ordering

# DeltaMSqIJ = m^2_j - m^2_i

def reqparameters(ns, **kwargs):
    pdg_year = kwargs.get('pdg_year', 2016)
    curpdg = pdg[pdg_year]
    ns.reqparameter('SinSq12', central=curpdg['sinSqtheta12'],
                     sigma=curpdg['sinSqtheta12_e'], limits=(0, 1), label='Solar mixing angle sin²θ₁₂')

    ns.reqparameter('DeltaMSq12', central=curpdg['dmSq21'],
                      sigma=curpdg['dmSq21_e'], limits=(0, 0.1), label='Solar mass splitting |Δm²₂₁|')

    ns.reqparameter('SinSq13', central=curpdg['sinSqtheta13'],
                     sigma=curpdg['sinSqtheta13_e'], limits=(0, 1), label='Reactor mixing angle sin²θ₁₃ ')

    ns.reqparameter('Alpha', type='discrete', default='normal',
                     variants={'normal': 1.0, 'inverted': -1.0}, label='Neutrino mass ordering α')

    ns.reqparameter('SinSq23', central=curpdg['sinSqtheta23_normal'],
                      sigma=curpdg['sinSqtheta23_normal_e'], limits=(0, 1), label='Atmospheric mixing angle sin²θ₂₃')

    #  ns.reqparameter('DeltaMSq23', central=curpdg['dmSq32_normal'],
                     #  sigma=curpdg['dmSq32_normal_e'], limits=(0, 0.1))

    try:
        ns.reqparameter('DeltaMSqEE', central=curpdg['dmSqEE'],
                          sigma=curpdg['dmSqEE_e'], limits=(0, 0.1), label='Reactor average mass splitting |Δm²(ee)|')
    except KeyError:
        ns.reqparameter('DeltaMSq23', central=curpdg['dmSq32'],
                          sigma=curpdg['dmSq32_e'], limits=(0, 0.1), label='Mass splitting (2, 3)')



    ns.reqparameter('Delta', type='uniformangle', central=0.0, label='CP violation phase δ(CP)')
    ns.reqparameter("SigmaDecohRel", central=1.e-5, sigma=1e-5, label='Relative momentum spread (decoherence)')

    with ns:
        C.OscillationExpressions(ns=ns)
        C.PMNSExpressions(ns=ns)

def reqparameters_reactor(ns, dm, **kwargs):
    """Reactor oscillation parameters"""
    pdg_year = kwargs.get('pdg_year', 2016)
    curpdg = pdg[pdg_year]

    if dm=='23':
        ns.reqparameter('DeltaMSq23', central=curpdg['dmSq32_normal'],
                         sigma=curpdg['dmSq32_normal_e'], limits=(0, 0.1), label='Mass splitting |Δm²₂₃|')
    elif dm=='ee':
        ns.reqparameter('DeltaMSqEE', central=curpdg['dmSqEE'],
                          sigma=curpdg['dmSqEE_e'], limits=(0, 0.1), label='Reactor average mass splitting |Δm²(ee)|')
    else:
        raise Exception('Invalid Δm² definition: '+str(dm))

    ns.reqparameter('SinSqDouble13', central=curpdg['sinSq2theta13'],
                     sigma=curpdg['sinSq2theta13_e'], limits=(0, 1), label='Reactor mixing angle sin²2θ₁₃ ')

    ns.reqparameter('DeltaMSq12', central=curpdg['dmSq21'],
                      sigma=curpdg['dmSq21_e'], limits=(0, 0.1), label='Solar mass splitting |Δm²₂₁|')

    ns.reqparameter('SinSqDouble12', central=curpdg['sinSq2theta12'],
                     sigma=curpdg['sinSq2theta12_e'], limits=(0, 1), label='Solar mixing angle sin²2θ₁₂')

    ns.reqparameter('Alpha', type='discrete', default='normal',
                     variants={'normal': 1.0, 'inverted': -1.0}, label='Neutrino mass ordering α')

    ns.reqparameter('SinSq23', central=curpdg['sinSqtheta23_normal'],
                      sigma=curpdg['sinSqtheta23_normal_e'], limits=(0, 1), label='Atmospheric mixing angle sin²θ₂₃')

    ns.reqparameter('Delta', type='uniformangle', central=0.0, label='CP violation phase δ(CP)')
    ns.reqparameter("SigmaDecohRel", central=1.e-5, sigma=1e-5, label='Relative momentum spread (decoherence)')

    with ns:
        C.OscillationExpressions(ns=ns)
        C.PMNSExpressionsC(ns=ns)
