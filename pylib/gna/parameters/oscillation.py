import ROOT
from gna.env import env

def reqparameters(ns):
    ns.reqparameter('SinSq12', central=0.307, sigma=0.017, limits=(0, 1))
    ns.reqparameter('SinSq13', central=0.0214, sigma=0.002, limits=(0, 1))
    ns.reqparameter('SinSq23', central=0.42, sigma=0.08, limits=(0, 1))

    ns.reqparameter('DeltaMSq12', central=7.54e-5, sigma=0.24e-5,
                    limits=(0, 0.1))
    ns.reqparameter('DeltaMSqEE', central=2.34e-3, sigma=0.10e-3,
                    limits=(0, 0.1))
    ns.reqparameter('Alpha', type='discrete', default='normal',
                    variants={'normal': 1.0, 'inverted': -1.0})

    ns.reqparameter('Delta', type='uniformangle', central=0.0)
    ns.reqparameter("SigmaDecohRel", central=1.e-5, sigma=0)
    with ns:
        ROOT.OscillationExpressions(ns=ns)
        ROOT.PMNSExpressions(ns=ns)
