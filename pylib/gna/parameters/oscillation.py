import ROOT

def defparameters(ns):
    ns.defparameter('SinSq12', central=0.307, sigma=0.017, limits=(0, 1))
    ns.defparameter('SinSq13', central=0.0214, sigma=0.002, limits=(0, 1))
    ns.defparameter('SinSq23', central=0.42, sigma=0.08, limits=(0, 1))

    ns.defparameter('DeltaMSq12', central=7.54e-5, sigma=0.24e-5,
                    limits=(0, 0.1))
    ns.defparameter('DeltaMSqEE', central=2.44e-3, sigma=0.10e-3,
                    limits=(0, 0.1))
    ns.defparameter('Alpha', type='discrete', default='normal',
                    variants={'normal': 1.0, 'inverted': -1.0})

    ns.defparameter('Delta', type='uniformangle', central=0.0)

    with ns:
        ROOT.OscillationExpressions(ns=ns)
        ROOT.PMNSExpressions(ns=ns)
