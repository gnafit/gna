#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import NestedDict
from collections import OrderedDict
from physlib import percent

class uncertain(object):
    def __init__(self, central, uncertainty, mode):
        assert mode in ['absolute', 'relative', 'percent'], 'Unsupported uncertainty mode '+mode

        if mode=='percent':
            mode='relative'
            uncertainty*=0.01

        if mode=='relative':
            assert central!=0, 'Central value should differ from 0 for relative uncertainty'

        self.central = central
        self.uncertainty   = uncertainty
        self.mode    = mode

    def __str__(self):
        res = '{central:.6g}'.format(central=self.central)

        if self.mode=='relative':
            sigma    = self.central*self.uncertainty
            relsigma = self.uncertainty
        else:
            sigma    = self.uncertainty
            relsigma = sigma/self.central

        res +=( '±{sigma:.6g}'.format(sigma=sigma) )

        if self.central:
            res+=( ' [{relsigma:.6g}%]'.format(relsigma=relsigma*100.0) )

        return res

    def __repr__(self):
        return 'uncertain({central!r}, {uncertainty!r}, {mode!r})'.format( **self.__dict__ )

def uncertaindict(*args, **kwargs):
    common = dict(
        central = kwargs.pop( 'central', None ),
        uncertainty   = kwargs.pop( 'uncertainty',   None ),
        mode    = kwargs.pop( 'mode',    None ),
    )
    missing = [ s for s in ['central', 'uncertainty', 'mode'] if not common[s] ]
    res  = OrderedDict( *args, **kwargs )

    for k, v in res.items():
        kcommon = common.copy()
        if isinstance(v, dict):
            kcommon.update( v )
        else:
            if isinstance( v, (int, float) ):
                v = (v, )
            kcommon.update( zip( missing, v ) )

        res[k] = uncertain( **kcommon )

    return res

cfg = NestedDict()
cfg.detectors = [ 'AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34' ]
cfg.groups    = OrderedDict( [
                ( 'EH1', ('AD11', 'AD12') ),
                ( 'EH2', ('AD21', 'AD22') ),
                ( 'EH3', ('AD31', 'AD32', 'AD33', 'AD34') ),
                ] )

bkg = cfg('bkg')
bkg.list = [ 'acc', 'lihe', 'fastn', 'amc', 'alphan' ]

bkg.acc = NestedDict(
        norm = uncertain( 1.0, 1.0, 'percent' ),
        spectra = NestedDict(
            bundle = 'root_histograms_v01',
            file   = 'FIXME: data/light_events/dubna/accspec_scaled_all.root',
            format = '{group}_{det}_singleTrigEnergy',
            grouping = 'individual',
            )
        )

cfg.lihe = NestedDict(
        rates = NestedDict(
            docdb = [10956],
            lifraction = uncertain( 0.95, 0.05, 'percent' ),
            correlation = 'group',
            rates = uncertaindict(
                mode = 'absolute',
                EH1 = (2.71, 0.90),
                EH2 = (1.91, 0.73),
                EH3 = (0.22, 0.07),
                )
            ),
        spectra = NestedDict(
            docdb = [8772, 8860],
            list = [ 'li_spectrum', 'he_spectrum' ],
            li_spectrum = NestedDict(
                bundle = 'root_histograms_v01',
                file   = 'data/background/lihe/13.09/toyli9spec_BCWmodel_v1.root',
                format = 'h_eVisAllSmeared',
                grouping = 'all',
                ),
            he_spectrum = NestedDict(
                bundle = 'root_histograms_v01',
                file   = 'data/background/lihe/13.09/toyhe8spec_BCWmodel_v1.root',
                format = 'h_eVisAllSmeared',
                grouping = 'all',
                )
            )
        )

cfg.fastn = NestedDict(
        docdb = [10948],
        rates = uncertaindict(
            mode = 'absolute',
            EH1  = (0.843, 0.083),
            EH2  = (0.638, 0.062),
            EH3  = (0.05 , 0.009),
            ),
        spectra = NestedDict(
            mode = 'IntegratorFunToHist1', #FIXME
            fun  = '(x/[0])^(-x/[0])',     #FIXME
            range=(0.7, 12.0),
            pars=NestedDict(
                values = uncertaindict(
                    mode='relative',
                    EH1=(67.79, 0.1132),
                    EH2=(58.30, 0.0817),
                    EH3=(68.02, 0.0997),
                    )
                ),
            )
        )

cfg.amc = NestedDict(
        docdb=[10956],
        rates = uncertaindict(
            mode='absolute',
            AD11 = (0.18, 0.08),
            AD12 = (0.18, 0.08),
            AD21 = (0.16, 0.07),
            AD22 = (0.15, 0.07),
            AD31 = (0.07, 0.03),
            AD32 = (0.06, 0.03),
            AD33 = (0.07, 0.03),
            AD34 = (0.05, 0.02)
            ),
        spectra = NestedDict(
            bundle = 'root_histograms_v01',
            file = 'data/background/amc/13.09/p12b_amc_fit.root',
            format = 'hCorrAmCPromptSpec',
            grouping = 'all',
            )
        )

cfg.alphan = NestedDict(
        docdb=[10956],
        rates = uncertaindict(
            mode='relative',
            uncertainty=50.0*percent,
            AD11 = 0.08,
            AD12 = 0.07,
            AD21 = 0.05,
            AD22 = 0.07,
            AD31 = 0.05,
            AD32 = 0.05,
            AD33 = 0.05,
            AD34 = 0.05
            ),
        spectra = NestedDict(
            bundle = 'root_histograms_v01',
            file = 'data/background/alpha_n/13.09/p12b_alpha_n.root',
            format = '{detshort}',
            grouping = 'individual',
            )
        )

print(str(cfg))

exec 'cfg_clone='+repr(cfg)
clonable = str(cfg)==str(cfg_clone)
print( '\033[32mNestedDict is clonable!' if clonable else '\033[31mNestedDict clone FAIL!', '\033[0m' )

