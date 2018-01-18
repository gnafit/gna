#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import NestedDict
from collections import OrderedDict
from physlib import percent

class uncertain(object):
    def __init__(self, central, error, mode):
        assert mode in ['absolute', 'relative', 'percent'], 'Unsupported uncertainty mode '+mode

        if mode=='percent':
            mode='relative'
            error*=0.01

        if mode=='relative':
            assert central!=0, 'Central value should differ from 0 for relative uncertainty'

        self.central = central
        self.error   = error
        self.mode    = mode

    def __str__(self):
        res = '{central:.6g}'.format(central=self.central)

        if self.mode=='relative':
            sigma    = self.central*self.error
            relsigma = self.error
        else:
            sigma    = self.error
            relsigma = sigma/self.central

        res +=( '±{sigma:.6g}'.format(sigma=sigma) )

        if self.central:
            res+=( ' [{relsigma:.6g}%]'.format(relsigma=relsigma*100.0) )

        return res

    def __repr__(self):
        return 'uncertain({central!r}, {error!r}, {mode!r})'.format( **self.__dict__ )

def uncertaindict(*args, **kwargs):
    mode = kwargs.pop( 'mode' )
    res  = OrderedDict( *args, **kwargs )

    for k in res.keys():
        res[k] = uncertain( *res[k], mode=mode )

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
            file   = 'FIXME: data/light_events/dubna/accspec_scaled_all.root',
            format = 'EH{site}_AD{det}_singleTrigEnergy'
            )
        )

cfg.lihe = NestedDict(
        rates = NestedDict(
            docdb = [10956],
            lifraction = uncertain( 0.95, 0.05, 'percent' ),
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
                file   = 'data/background/lihe/13.09/toyli9spec_BCWmodel_v1.root',
                format = 'h_eVisAllSmeared'
                ),
            he_spectrum = NestedDict(
                file   = 'data/background/lihe/13.09/toyhe8spec_BCWmodel_v1.root',
                format = 'h_eVisAllSmeared'
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
            mode = 'IntegratorFunToHist1', # FIXME
            fun  = '(x/[0])^(-x/[0])',     # FIXME
            range=(0.7, 12.0),
            pars=NestedDict(
                correlation='site',
                values = uncertaindict(
                    mode='relative',
                    EH1=(67.79, 0.1132),
                    EH2=(58.30, 0.0817),
                    EH3=(68.02, 0.0997),
                    )
                ),
            )
        )

# alphan     = 50.0*percent,
    # fastn      = dict( EH1=0.083, EH2=0.062, EH3=0.009, mode='absolute' ),
    # fastn_ihep = dict( EH1=0.103, EH2=0.074, EH3=0.009, mode='absolute' ),
    # lihe_ihep  = dict( EH1=1.06,  EH2=0.77,  EH3=0.06,  mode='absolute' ),
    # amc        = dict( AD11=0.08, AD12=0.08,
                       # AD21=0.07, AD22=0.07,
                       # AD31=0.03, AD32=0.03, AD33=0.03, AD34=0.02,
                       # mode='absolute'
                       # ),
    # amc6ad     = dict( AD11=0.12, AD12=0.11,
                       # AD21=0.13, #AD22=0.00,
                       # AD31=0.10, AD32=0.10, AD33=0.10, #AD34=0.00,
                       # mode='absolute'
                       # ),
    # amc8ad     = dict( AD11=0.07, AD12=0.06,
                       # AD21=0.06, AD22=0.07,
                       # AD31=0.02, AD32=0.02, AD33=0.02, AD34=0.02,
                       # mode='absolute'
                       # ),
    # amcp14a8ad = dict( AD11=0.09, AD12=0.10,
                       # AD21=0.08, AD22=0.10,
                       # AD31=0.03, AD32=0.02, AD33=0.02, AD34=0.03,
                       # mode='absolute'
                       # ),
    # amcp14a    = dict( AD11=0.10, AD12=0.10,
                       # AD21=0.09, AD22=0.10,
                       # AD31=0.05, AD32=0.04, AD33=0.04, AD34=0.03,
                       # mode='absolute'
                       # ),

print(str(cfg))

exec 'cfg_clone='+repr(cfg)
check = str(cfg)==str(cfg_clone)
print( '\033[32mNestedDict is clonable!' if else '\033[31mNestedDict clone FAIL!', '\033[0m' )
