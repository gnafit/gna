#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import NestedDict
from collections import OrderedDict
from physlib import percent

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
            file   = 'data/dayabay/data_spectra/P15A_IHEP_data/data_IHEP_P15A_All.root',
            format = 'accidental_pred_{}',
            variants = OrderedDict([
                ( 'AD11', 'EH1_AD1' ), ( 'AD12', 'EH1_AD2' ),
                ( 'AD21', 'EH2_AD3' ), ( 'AD22', 'EH2_AD8' ),
                ( 'AD31', 'EH3_AD4' ), ( 'AD32', 'EH3_AD5' ), ( 'AD33', 'EH3_AD6' ), ( 'AD34', 'EH3_AD7' ),
                ])
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
                ),
            he_spectrum = NestedDict(
                bundle = 'root_histograms_v01',
                file   = 'data/background/lihe/13.09/toyhe8spec_BCWmodel_v1.root',
                format = 'h_eVisAllSmeared',
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
            )
        )

print(str(cfg))

exec 'cfg_clone='+repr(cfg)
clonable = str(cfg)==str(cfg_clone)
print( '\033[32mNestedDict is clonable!' if clonable else '\033[31mNestedDict clone FAIL!', '\033[0m' )

