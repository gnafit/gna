import ROOT
from minimize.lib.minuit_base import MinuitBase

TMinuit = ROOT.TMinuitMinimizer

ROOT.TMinuitMinimizer.UseStaticMinuit(False)
class Minuit(TMinuit, MinuitBase):
    _label = 'TMinuit'
    def __init__(self, statistic, minpars, **kwargs):
        TMinuit.__init__(self)
        MinuitBase.__init__(self, statistic, minpars, **kwargs)
