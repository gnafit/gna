# -*- coding: utf-8 -*-
import ROOT
from packages.minimize.lib.minuit_base import MinuitBase

TMinuit = ROOT.TMinuitMinimizer

class Minuit(TMinuit, MinuitBase):
    _label = 'TMinuit'
    def __init__(self, statistic, minpars, **kwargs):
        TMinuit.__init__(self)
        MinuitBase.__init__(self, statistic, minpars, **kwargs)

        ROOT.TMinuitMinimizer.UseStaticMinuit(False)
