import ROOT
from packages.minimize.lib.minuit_base import MinuitBase

TMinuit2 = ROOT.Minuit2.Minuit2Minimizer

class Minuit2(TMinuit2, MinuitBase):
    _label = 'TMinuit2'
    def __init__(self, statistic, minpars, **kwargs):
        TMinuit2.__init__(self)
        MinuitBase.__init__(self, statistic, minpars, **kwargs)
